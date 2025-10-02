import argparse
import os
import shutil
import time
from typing import List, Optional

import torch
import yaml
from loguru import logger
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import helpers.custom_types as custom_types
import metrics
import wandb
from helpers.factory import get_dataset, get_loss_function, get_metrics, get_model
from helpers.utils import drop_condition, save_eval_results


# -----------------------------
# Training one epoch
# -----------------------------
def train_one_epoch(
    model: custom_types.GenBaseModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: custom_types.DeviceType,
    epoch: int = 0,
    scaler: Optional[GradScaler] = None,
) -> torch.Tensor:
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Use labels only if model can do conditioning
        conditioning = None
        if model.has_conditional_generation:
            conditioning = labels

        # forward pass with AMP
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs, conditioning=conditioning)
            loss = criterion(outputs, inputs)

        # backward pass with AMP scaler
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        wandb.log({"batch_loss": loss.item()}, step=epoch * len(dataloader) + idx)
        pbar.set_postfix({"loss": total_loss / (idx + 1)})

    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
    return avg_loss


# -----------------------------
# Evaluation (sampling)
# -----------------------------
def eval_sample(
    model: custom_types.GenBaseModel,
    num_samples: int,
    device: custom_types.DeviceType,
    img_size: int,
    step: int = 0,
    save_dir: str = "./",
    dataloader: Optional[torch.utils.data.DataLoader] = None,
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
            conditioning = drop_condition(conditioning, 0.25)
            samples = model.sample(
                num_samples,
                device,
                img_size,
                batch_size=num_samples,
                conditioning=conditioning,
            )
        else:
            # unconditional sampling
            samples = model.sample(
                num_samples, device, img_size, batch_size=num_samples
            )

    # log a few images
    samples_cpu = samples.cpu()
    for idx in range(min(num_samples, 4)):
        wandb.log({f"generated_{idx}": wandb.Image(samples_cpu[idx])}, step=step)

    save_eval_results(
        samples_cpu,
        filename=f"{save_dir}/generated_samples_step_{step}.png",
        conditioning=conditioning,
    )
    return samples


# -----------------------------
# Training loop
# -----------------------------
def train(
    model: custom_types.GenBaseModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    compute_metrics: List[torch.nn.Module],
    device: custom_types.DeviceType,
    epochs: int = 10,
    start_epoch: int = 0,
    metric_interval: int = 1,
    save_dir: str = "./",
):
    model.to(device)
    scaler = GradScaler()  # AMP gradient scaler
    prev_best_score = float("inf")

    for epoch in range(start_epoch, epochs):
        epoch_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device, epoch, scaler
        )
        wandb.log({"epoch_loss": epoch_loss}, step=(epoch + 1) * len(dataloader))

        # Generate evaluation samples/metrics
        if (epoch + 1) % metric_interval == 0 or epoch + 1 == epochs:
            eval_sample(
                model,
                num_samples=16,
                device=device,
                img_size=dataloader.img_size,
                step=(epoch + 1) * len(dataloader),
                save_dir=save_dir,
                dataloader=dataloader,
            )
            curr_score = prev_best_score
            for metric in compute_metrics:
                # FID is computed on unconditioned input only
                if type(metric) == metrics.FIDInception:
                    sampler_loader = model.wrap_sampler_to_loader(
                        num_samples=metric.samples,
                        device=device,
                        img_size=dataloader.img_size,
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
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "fid": curr_score,
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

    # Setup loss
    criterion = get_loss_function(cfg["loss"])
    # Setup Optimizer
    if cfg["optimizer"]["type"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["lr"])
    else:
        raise ValueError(f"Unsupported optimizer {cfg['optimizer']['type']}")

    # Load checkpoint if provided
    start_epoch = 0
    if cfg["model"].get("checkpoint", None) is not None:
        ckpt = cfg["model"]["checkpoint"]
        checkpoint = torch.load(ckpt, map_location=cfg["training"]["device"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(cfg["training"]["device"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Loaded model checkpoint from {ckpt}")

    # Metrics
    compute_metrics = get_metrics(cfg.get("metrics", None))

    # Training loop
    train(
        model,
        dataloader,
        optimizer,
        criterion,
        compute_metrics,
        metric_interval=cfg["training"]["metric_interval"],
        device=cfg["training"]["device"],
        epochs=cfg["training"]["epochs"],
        start_epoch=start_epoch,
        save_dir=f"./runs/{run_name}",
    )


if __name__ == "__main__":
    main()
