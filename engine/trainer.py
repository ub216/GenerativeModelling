import contextlib
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
from loguru import logger
from torch.amp import autocast
from tqdm import tqdm

import helpers.custom_types as custom_types
import helpers.distributed_utils as dist_utils
import metrics
import wandb
from engine.optimizer_manager import OptimizerManager
from helpers.diffusion_utils import drop_condition
from helpers.utils import save_eval_results


# -----------------------------
# Training one epoch
# -----------------------------
def train_one_epoch(
    model: custom_types.GenBaseModel,
    model_ema: custom_types.GenEMAModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer_manager: OptimizerManager,
    criterion: torch.nn.Module,
    device: custom_types.DeviceType,
    epoch: int = 0,
    amp_dtype: torch.dtype | None = torch.float16,
) -> torch.Tensor:
    t0 = time.time()
    model.train()
    total_loss = defaultdict(float)

    # Shuffle data differently at each epoch for distributed training
    if dist_utils.is_distributed() and hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False, disable=not dist_utils.is_main_process())
    optimizer_manager.zero_grad()

    for idx, (inputs, labels) in enumerate(pbar):

        inputs = inputs.to(device, non_blocking=True)

        # Use labels only if model can do conditioning
        conditioning = None
        if model.has_conditional_generation:
            conditioning = labels

        # forward pass with AMP
        amp_ctx = autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
        with amp_ctx:
            outputs = model(inputs, conditioning=conditioning)
            try:
                loss = criterion(outputs, inputs)
            except (ValueError, TypeError) as e:
                raise type(e)(f"Loss {criterion.__class__.__name__} failed at epoch {epoch}, step {idx}: {e}") from e

        # Wrap loss in dict if not already
        # This is needed for optimizer manager
        # to handle multiple optimizers (like GANs)
        if isinstance(loss, torch.Tensor):
            loss = {"all": loss}

        # backward pass with AMP scaler
        optimizer_manager.backward(loss, model=model)

        step_metrics = optimizer_manager.step()
        if step_metrics:
            optimizer_manager.zero_grad()
            # Update EMA model
            model_ema.update(model)

        # Logging — emit batch losses and grad norms in a single call per step
        update_stats = {}
        step_log = {f"batch_{key}_loss": loss[key].item() for key in loss}
        if step_metrics:
            step_log.update(step_metrics)
        if dist_utils.is_main_process():
            wandb.log(step_log, step=epoch * len(dataloader) + idx)
        for key in loss.keys():
            total_loss[key] += loss[key].item()
            update_stats[key] = total_loss[key] / (idx + 1)
        pbar.set_postfix(update_stats)

    # Backpropogate all losses missed by accumulation step
    # The losses are incorrectly scaled when force updated
    # TODO: adjust this
    # optimizer_manager.step(force=True)
    epoch_time = time.time() - t0
    update_stats = f"Epoch {epoch+1}, Time: {epoch_time:.2f}s,"
    for key in total_loss.keys():
        avg_loss = total_loss[key] / len(dataloader)
        update_stats += f" {key} : {avg_loss:.4f}"
    logger.info(update_stats)
    return torch.tensor(avg_loss)


# -----------------------------
# Sample generation (no I/O)
# -----------------------------
def generate_samples(
    model: custom_types.GenBaseModel,
    num_samples: int,
    device: custom_types.DeviceType,
    image_size: int | Tuple[int, int],
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    num_fixed_samples: int = 0,
    fixed_seed: int = 42,
) -> Tuple[torch.Tensor, Optional[List]]:
    """Generate samples and return (samples_cpu, conditioning).

    All ranks must call this — with FSDP the forward pass issues allgather
    collectives that require every rank to participate. Logging and saving
    are the caller's responsibility.
    """
    model.eval()
    conditioning = None
    with torch.no_grad():
        if dataloader is not None and model.has_conditional_generation:
            conditioning = []
            for _, lbls in dataloader:
                conditioning.extend(lbls)
                if len(conditioning) >= num_samples:
                    break
            conditioning = conditioning[:num_samples]
            conditioning = (
                drop_condition(conditioning, 0.25) if sum([c == "" for c in conditioning]) == 0 else conditioning
            )

        # For reproducibility, generate a set of samples for the first `num_fixed_samples` using a fixed seed.
        n_fixed = min(num_fixed_samples, num_samples)
        if n_fixed > 0:
            device_obj = device if isinstance(device, torch.device) else torch.device(device)
            with torch.random.fork_rng(devices=[device_obj] if device_obj.type == "cuda" else []):
                torch.manual_seed(fixed_seed)
                samples = model.sample(
                    n_fixed,
                    device,
                    image_size,
                    batch_size=n_fixed,
                    conditioning=conditioning[:n_fixed] if conditioning else None,
                )
            if n_fixed < num_samples:
                rest = model.sample(
                    num_samples - n_fixed,
                    device,
                    image_size,
                    batch_size=num_samples - n_fixed,
                    conditioning=conditioning[n_fixed:] if conditioning else None,
                )
                samples = torch.cat([samples, rest], dim=0)
        else:
            samples = model.sample(num_samples, device, image_size, batch_size=num_samples, conditioning=conditioning)

    return samples.cpu(), conditioning


# -----------------------------
# Training loop
# -----------------------------
def train(
    model: custom_types.GenBaseModel,
    model_ema: custom_types.GenEMAModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer_manager: OptimizerManager,
    criterion: torch.nn.Module,
    compute_metrics: List[torch.nn.Module],
    device: custom_types.DeviceType,
    epochs: int = 10,
    start_epoch: int = 0,
    metric_interval: int | None = None,
    sample_interval: int = 1,
    save_dir: str = "./",
    save_after_epoch: int = float("inf"),
    amp_dtype: torch.dtype | None = torch.float16,
    num_fixed_samples: int = 0,
    fixed_seed: int = 42,
):
    prev_best_score = float("inf")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        epoch_loss = train_one_epoch(
            model, model_ema, dataloader, optimizer_manager, criterion, device, epoch, amp_dtype
        )
        if dist_utils.is_distributed():
            dist_utils.reduce_tensor(epoch_loss)
        epoch_time = time.time() - t0

        # Logging, sample generation, and checkpointing (rank 0 only)
        if dist_utils.is_main_process():
            wandb.log({"epoch_time": epoch_time, "epoch_loss": epoch_loss}, step=(epoch + 1) * len(dataloader))

            # Generate samples for monitoring (cheaper than FID; gated independently)
            normal_samples, ema_samples, sample_conditioning = None, None, None
            if (epoch + 1) % sample_interval == 0 or epoch + 1 == epochs:
                normal_samples, sample_conditioning = generate_samples(
                    dist_utils.unwrap_model(model),
                    num_samples=9,
                    device=device,
                    image_size=dataloader.image_size,
                    dataloader=dataloader,
                    num_fixed_samples=num_fixed_samples,
                    fixed_seed=fixed_seed,
                )
                ema_samples, _ = generate_samples(
                    dist_utils.unwrap_model(model_ema),
                    num_samples=9,
                    device=device,
                    image_size=dataloader.image_size,
                    dataloader=dataloader,
                    num_fixed_samples=num_fixed_samples,
                    fixed_seed=fixed_seed,
                )

            if normal_samples is not None:
                # Log a few EMA images and save grids for both normal and EMA models
                for idx in range(min(9, 4)):
                    wandb.log(
                        {f"ema_generated_{idx}": wandb.Image(ema_samples[idx])}, step=(epoch + 1) * len(dataloader)
                    )
                save_eval_results(
                    normal_samples,
                    filename=f"{save_dir}/generated_samples_step_{(epoch + 1) * len(dataloader)}_normal.png",
                    conditioning=sample_conditioning,
                )
                save_eval_results(
                    ema_samples,
                    filename=f"{save_dir}/generated_samples_step_{(epoch + 1) * len(dataloader)}_ema.png",
                    conditioning=sample_conditioning,
                )

            # Compute evaluation metrics
            scores = {}
            curr_score = None
            if metric_interval is not None and ((epoch + 1) % metric_interval == 0 or epoch + 1 == epochs):
                for metric in compute_metrics:
                    # metrics are computed on unconditioned input only
                    if isinstance(metric, metrics.ImageDistributionMetric):
                        sampler_loader = dist_utils.unwrap_model(model).wrap_sampler_to_loader(
                            num_samples=metric.samples,
                            device=device,
                            image_size=dataloader.image_size,
                            batch_size=dataloader.batch_size,
                        )
                        score = metric(dataloader, sampler_loader)
                        scores[metric.name] = score
                        if metric.primary_metric:
                            curr_score = score

            if scores:
                wandb.log(scores, step=(epoch + 1) * len(dataloader))
                logger.info(f"Epoch {epoch+1}: " + ", ".join(f"{k}: {v:.4f}" for k, v in scores.items()))
                # Save best model
                if curr_score is not None and (curr_score <= prev_best_score or epoch + 1 == epochs):
                    prev_best_score = curr_score
                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": dist_utils.get_state_dict(model),
                        "model_ema_state_dict": dist_utils.get_state_dict(model_ema),
                        "optimizer_state_dict": optimizer_manager.state_dict(),
                        "loss": epoch_loss,
                        "primary_score": curr_score,
                    }
                    torch.save(checkpoint, f"{save_dir}/best_{epoch+1}.pth")

            if epoch % save_after_epoch == 0:
                # Save last model
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": dist_utils.get_state_dict(model),
                    "model_ema_state_dict": dist_utils.get_state_dict(model_ema),
                    "optimizer_state_dict": optimizer_manager.state_dict(),
                    "loss": epoch_loss,
                }
                torch.save(checkpoint, f"{save_dir}/epoch_{epoch+1}.pth")

        if dist_utils.is_distributed():
            torch.distributed.barrier()  # sync before next epoch starts
    logger.info("Training complete.")
