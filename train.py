import argparse
import os
import shutil
import time

import torch
import yaml
from loguru import logger
from torch.amp import GradScaler, autocast
from tqdm import tqdm

import metrics
import wandb
from factory import get_dataset, get_loss_function, get_metrics, get_model
from utils import save_eval_results


# -----------------------------
# Training one epoch
# -----------------------------
def train_one_epoch(
    model, dataloader, optimizer, criterion, device, epoch=0, scaler=None
):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for idx, (inputs, _) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # forward pass with AMP
        with autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
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
def eval_sample(model, num_samples, device, img_size, step=0, save_dir="./"):
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device, img_size, batch_size=num_samples)
    samples_cpu = samples.cpu()
    for idx in range(min(num_samples, 4)):
        wandb.log({f"generated_{idx}": wandb.Image(samples_cpu[idx])}, step=step)
    save_eval_results(
        samples_cpu, filename=f"{save_dir}/generated_samples_step_{step}.png"
    )
    return samples


# -----------------------------
# Training loop
# -----------------------------
def train(
    model,
    dataloader,
    optimizer,
    criterion,
    compute_metrics,
    device,
    epochs=10,
    start_epoch=0,
    metric_interval=1,
    save_dir="./",
):
    model.to(device)
    scaler = GradScaler()  # AMP gradient scaler
    prev_best_score = float("inf")

    for epoch in range(start_epoch, epochs):
        epoch_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, device, epoch, scaler
        )
        wandb.log({"epoch_loss": epoch_loss}, step=(epoch + 1) * len(dataloader))

        # Generate evaluation samples
        if (epoch + 1) % metric_interval == 0 or epoch + 1 == epochs:
            eval_sample(
                model,
                num_samples=16,
                device=device,
                img_size=dataloader.img_size,
                step=(epoch + 1) * len(dataloader),
                save_dir=save_dir,
            )

            for metric in compute_metrics:
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

            # Save best model
            if curr_score < prev_best_score or epoch + 1 == epochs:
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
def main(config_path="config.yaml"):
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
    os.makedirs(f"./runs/{run_name}", exist_ok=True)
    shutil.copy(config_path, f"./runs/{run_name}/config.yaml")

    # Setup model
    model = get_model(cfg["model"])

    # Setup loss
    criterion = get_loss_function(cfg["loss"])

    # Setup Dataloader
    dataloader = get_dataset(cfg["dataset"], cfg["training"].get("batch_size", None))

    # Setup Optimizer
    if cfg["optimizer"]["type"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["lr"])
    else:
        raise ValueError(f"Unsupported optimizer {cfg['optimizer']['type']}")

    # Load checkpoint if provided
    start_epoch = 0
    if cfg["model"].get("checkpoint", None) is not None:
        ckpt = cfg['model']['checkpoint']
        logger.info(f"Loaded model checkpoint from {ckpt}")
        checkpoint = torch.load(
            ckpt, map_location=cfg["training"]["device"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(cfg["training"]["device"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Metrics
    compute_metrics = get_metrics(cfg["metrics"])

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
