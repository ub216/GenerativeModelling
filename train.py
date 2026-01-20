import argparse
import os
import shutil
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import yaml
from loguru import logger
from torch.amp import autocast
from tqdm import tqdm

import helpers.custom_types as custom_types
import metrics
import wandb
from helpers.factory import (get_dataset, get_loss_function, get_metrics,
                             get_model, get_optimizer_manager)
from helpers.optimizer_manager import OptimizerManager
from helpers.utils import drop_condition, save_eval_results
from models.ema import EMAModel


# -----------------------------
# Training one epoch
# -----------------------------
def train_one_epoch(
    model: custom_types.GenBaseModel,
    model_ema: EMAModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer_manager: OptimizerManager,
    criterion: torch.nn.Module,
    device: custom_types.DeviceType,
    epoch: int = 0,
) -> torch.Tensor:
    t0 = time.time()
    model.train()
    total_loss = defaultdict(float)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for idx, (inputs, labels) in enumerate(pbar):
        optimizer_manager.zero_grad()

        inputs = inputs.to(device, non_blocking=True)

        # Use labels only if model can do conditioning
        conditioning = None
        if model.has_conditional_generation:
            conditioning = labels

        # forward pass with AMP
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs, conditioning=conditioning)
            loss = criterion(outputs, inputs)

        # Wrap loss in dict if not already
        # This is needed for optimizer manager
        # to handle multiple optimizers (like GANs)
        if isinstance(loss, torch.Tensor):
            loss = {"all": loss}

        # backward pass with AMP scaler
        optimizer_manager.backward(loss)

        metrics = optimizer_manager.step()
        if metrics:
            wandb.log(metrics, step=epoch * len(dataloader) + idx)
            # Update EMA model
            model_ema.update(model)

        # Logging
        update_stats = {}
        for key in loss.keys():
            total_loss[key] += loss[key].item()
            wandb.log(
                {f"batch_{key}_loss": loss[key].item()},
                step=epoch * len(dataloader) + idx,
            )
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
    return avg_loss


# -----------------------------
# Evaluation (sampling)
# -----------------------------
def eval_sample(
    model: custom_types.GenBaseModel,
    num_samples: int,
    device: custom_types.DeviceType,
    image_size: int | Tuple[int, int],
    step: int = 0,
    save_dir: str = "./",
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    is_ema: bool = True,
) -> torch.Tensor:
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
                drop_condition(conditioning, 0.25)
                if sum([c == "" for c in conditioning]) == 0
                else conditioning
            )
            samples = model.sample(
                num_samples,
                device,
                image_size,
                batch_size=num_samples,
                conditioning=conditioning,
            )
        else:
            # unconditional sampling
            samples = model.sample(
                num_samples, device, image_size, batch_size=num_samples
            )

    # log a few images
    samples_cpu = samples.cpu()
    if is_ema:
        for idx in range(min(num_samples, 4)):
            wandb.log({f"generated_{idx}": wandb.Image(samples_cpu[idx])}, step=step)

    save_eval_results(
        samples_cpu,
        filename=f"{save_dir}/generated_samples_step_{step}_{'ema' if is_ema else 'normal'}.png",
        conditioning=conditioning,
    )
    return samples


# -----------------------------
# Training loop
# -----------------------------
def train(
    model: custom_types.GenBaseModel,
    model_ema: EMAModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer_manager: OptimizerManager,
    criterion: torch.nn.Module,
    compute_metrics: List[torch.nn.Module],
    device: custom_types.DeviceType,
    epochs: int = 10,
    start_epoch: int = 0,
    metric_interval: int = 1,
    save_dir: str = "./",
    save_after_epoch: int = float("inf"),
):
    prev_best_score = float("inf")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        epoch_loss = train_one_epoch(
            model, model_ema, dataloader, optimizer_manager, criterion, device, epoch
        )
        epoch_time = time.time() - t0
        wandb.log({"epoch_time": epoch_time}, step=(epoch + 1) * len(dataloader))
        wandb.log({"epoch_loss": epoch_loss}, step=(epoch + 1) * len(dataloader))

        # Generate samples for monitoring
        eval_sample(
            model,
            num_samples=16,
            device=device,
            image_size=dataloader.image_size,
            step=(epoch + 1) * len(dataloader),
            save_dir=save_dir,
            dataloader=dataloader,
            is_ema=False,
        )
        eval_sample(
            model_ema,
            num_samples=16,
            device=device,
            image_size=dataloader.image_size,
            step=(epoch + 1) * len(dataloader),
            save_dir=save_dir,
            dataloader=dataloader,
        )
        # Generate evaluation samples/metrics
        if (epoch + 1) % metric_interval == 0 or epoch + 1 == epochs:

            curr_score = prev_best_score
            for metric in compute_metrics:
                # FID is computed on unconditioned input only
                if type(metric) == metrics.FIDInception:
                    sampler_loader = model.wrap_sampler_to_loader(
                        num_samples=metric.samples,
                        device=device,
                        image_size=dataloader.image_size,
                        batch_size=dataloader.batch_size,
                    )
                    curr_score = metric(dataloader, sampler_loader)
                    wandb.log({"FID": curr_score}, step=(epoch + 1) * len(dataloader))
                    logger.info(f"Epoch {epoch+1} FID: {curr_score:.4f}")

            # Save best and last model
            if curr_score <= prev_best_score or epoch + 1 == epochs:
                prev_best_score = curr_score
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "model_ema_state_dict": model_ema.state_dict(),
                    "optimizer_state_dict": optimizer_manager.state_dict(),
                    "loss": epoch_loss,
                    "fid": curr_score,
                }
                torch.save(checkpoint, f"{save_dir}/best_{epoch+1}.pth")
        if save_after_epoch <= epoch:
            # Save last model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_ema_state_dict": model_ema.state_dict(),
                "optimizer_state_dict": optimizer_manager.state_dict(),
                "loss": epoch_loss,
            }
            torch.save(checkpoint, f"{save_dir}/epoch_{epoch+1}.pth")
    logger.info("Training complete.")


# -----------------------------
# Main
# -----------------------------
def main(config_path: str = "config.yaml"):
    parser = argparse.ArgumentParser(description="Train generative model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    config_path = args.config
    assert os.path.isfile(config_path), f"Config file {config_path} not found"

    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Enable cuDNN optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Setup experiment
    torch.manual_seed(cfg["experiment"].get("seed", 42))
    logger.info(f"Torch seed set to {cfg['experiment'].get('seed', 42)}")
    run_name = f"{cfg['model']['type'].lower()}_{cfg['dataset']['type'].lower()}_{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=cfg["experiment"]["name"],
        name=run_name,
        config=cfg,
    )
    run_dir = f"./runs/{run_name}"
    logger.info(f"Saving intermidiate results to {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, f"./runs/{run_name}/config.yaml")

    # Setup Dataloader
    dataloader = get_dataset(cfg["dataset"], cfg["training"].get("batch_size", None))

    # Setup model after dataloader to estimate image_size and
    # channel dimension. This is required to initiate models
    model = get_model(cfg["model"], dataloader)

    # Create EMA model for stable sampling
    model_ema = EMAModel(
        model,
        decay=cfg["training"].get("ema_decay", 0),  # 0 means no EMA
    )

    model.to(cfg["training"]["device"])
    model_ema.to(cfg["training"]["device"])

    # Setup loss
    criterion = get_loss_function(cfg["loss"])

    # Setup Optimizer. This is done after the model is intialized
    optimizer_manager = get_optimizer_manager(cfg["optimizer"], model)

    # Load checkpoint if provided
    start_epoch = 0
    if cfg["model"].get("checkpoint", None) is not None:
        ckpt = cfg["model"]["checkpoint"]
        checkpoint = torch.load(ckpt, map_location=cfg["training"]["device"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model_ema.load_state_dict(checkpoint["model_ema_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer_manager.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Loaded model checkpoint from {ckpt}")

    # Metrics
    compute_metrics = get_metrics(cfg.get("metrics", None))

    # Training loop
    train(
        model,
        model_ema,
        dataloader,
        optimizer_manager,
        criterion,
        compute_metrics,
        metric_interval=cfg["training"]["metric_interval"],
        device=cfg["training"]["device"],
        epochs=cfg["training"]["epochs"],
        start_epoch=start_epoch,
        save_dir=f"./runs/{run_name}",
        save_after_epoch=cfg["training"].get("save_after_epoch", float("inf")),
    )


if __name__ == "__main__":
    main()
