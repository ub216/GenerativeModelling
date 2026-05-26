import argparse
import os
import shutil
import time

import torch
import yaml
from loguru import logger

from engine.evaluator import eval
from engine.factory import get_dataset, get_metrics, get_model


# -----------------------------
# Main
# -----------------------------
def main(config_path: str = "config.yaml"):
    parser = argparse.ArgumentParser(description="Evaluate generative model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--image_size", type=int, default=None, help="Model image size")
    parser.add_argument("--image_channels", type=int, default=None, help="Model image channels")
    args = parser.parse_args()
    config_path = args.config
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")

    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup experiment
    torch.manual_seed(cfg["experiment"].get("seed", 42))
    run_name = f"{cfg['model']['type'].lower()}_{cfg['dataset']['type'].lower()}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = f"./eval_runs/{run_name}"
    logger.info(f"Saving intermidiate results to {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, f"{run_dir}/config.yaml")

    # Setup model
    if args.image_size is not None and args.image_channels is not None:
        image_size = (args.image_size, args.image_size, args.image_channels)
        model, _ = get_model(cfg["model"], dataloader=None, image_size=image_size, build_ema=False)
    elif cfg["dataset"].get("type", None) is not None:
        dataloader = get_dataset(cfg["dataset"], cfg["training"].get("batch_size", None))
        model, _ = get_model(cfg["model"], dataloader=dataloader, build_ema=False)
    else:
        raise ValueError("Either dataset type or image_size and image_channels must be provided.")

    # Load checkpoint
    if cfg["model"].get("checkpoint", None) is not None:
        ckpt = cfg["model"]["checkpoint"]
        checkpoint = torch.load(ckpt, map_location=cfg["training"]["device"])
        if "model_ema_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_ema_state_dict"])
            logger.info("Loaded EMA weights for evaluation")
        elif "model_state_dict" in checkpoint:
            logger.info("Loaded model_state_dict weights for evaluation")
            model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model checkpoint from {ckpt}")
    else:
        logger.warning("No checkpoint provided, testing on random weights!")
    model = model.to(cfg["training"]["device"])

    # Metrics
    compute_metrics = get_metrics(cfg["metrics"])

    # n_samples
    n_samples = cfg.get("evaluation", {}).get("samples", 100)
    eval(
        model,
        dataloader,
        compute_metrics,
        device=cfg["training"]["device"],
        save_dir=f"{run_dir}",
        n_samples=n_samples,
    )


if __name__ == "__main__":
    main()
