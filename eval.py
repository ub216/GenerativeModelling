import argparse
import os
import shutil
import time

import torch
import yaml
from loguru import logger

import metrics
from factory import get_dataset, get_metrics, get_model
from utils import save_eval_results


# -----------------------------
# Evaluation loop
# -----------------------------
def eval_sample(model, num_samples, device, img_size, save_dir="./"):
    model.eval()
    with torch.no_grad():
        samples = model.sample(num_samples, device, img_size, batch_size=num_samples)
    samples_cpu = samples.cpu()
    save_eval_results(samples_cpu, filename=f"{save_dir}/generated_samples.png")
    return samples


def eval(model, dataloader, compute_metrics, device, save_dir="./"):
    # Generate evaluation samples
    eval_sample(
        model,
        num_samples=16,
        device=device,
        img_size=dataloader.img_size,
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
            logger.info(f"Final FID: {curr_score:.4f}")


# -----------------------------
# Main
# -----------------------------
def main(config_path="config.yaml"):
    parser = argparse.ArgumentParser(description="Evaluate generative model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    config_path = args.config
    assert os.path.isfile(config_path), f"Config file {config_path} not found"

    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup experiment
    run_name = f"{cfg['model']['type'].upper()}_{cfg['dataset']['type']}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = f"./eval_runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, f"{run_dir}/config.yaml")

    # Setup model
    model = get_model(cfg["model"])
    if cfg["model"].get("checkpoint", None) is not None:
        ckpt = cfg['model']['checkpoint']
        logger.info(f"Loaded model checkpoint from {ckpt}")
        checkpoint = torch.load(
            ckpt, map_location=cfg["training"]["device"]
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.warning("No checkpoint provided, testing on random weights!")
    model = model.to(cfg["training"]["device"])
    # Setup Dataloader
    dataloader = get_dataset(cfg["dataset"], cfg["training"].get("batch_size", None))

    # Metrics
    compute_metrics = get_metrics(cfg["metrics"])

    eval(model, dataloader, compute_metrics, device=cfg["training"]["device"], save_dir=f"{run_dir}")


if __name__ == "__main__":
    main()
