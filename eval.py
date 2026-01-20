import argparse
import os
import shutil
import time
from typing import List, Optional, Tuple

import torch
import yaml
from loguru import logger

import helpers.custom_types as custom_types
import metrics
from helpers.factory import get_dataset, get_metrics, get_model
from helpers.utils import drop_condition, save_eval_results


# -----------------------------
# Evaluation loop
# -----------------------------
def eval_sample(
    model: custom_types.GenBaseModel,
    num_samples: int,
    device: custom_types.DeviceType,
    image_size: int | Tuple[int, int],
    save_dir: str = "./",
    dataloader=None,
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
    filename = f"{save_dir}/generated_samples.png"
    save_eval_results(samples_cpu, filename=filename, conditioning=conditioning)
    logger.info(f"Saved results to {filename}")
    return samples


def eval(
    model: custom_types.GenBaseModel,
    dataloader: torch.utils.data.DataLoader,
    compute_metrics: List[torch.nn.Module],
    device: custom_types.DeviceType,
    save_dir: str = "./",
    n_samples: int = 100,
):
    # Generate evaluation samples
    eval_sample(
        model,
        num_samples=n_samples,
        device=device,
        image_size=dataloader.image_size,
        save_dir=save_dir,
        dataloader=dataloader,
    )
    for metric in compute_metrics:
        if type(metric) == metrics.FIDInception:
            sampler_loader = model.wrap_sampler_to_loader(
                num_samples=metric.samples,
                device=device,
                image_size=dataloader.image_size,
                batch_size=dataloader.batch_size,
            )
            curr_score = metric(dataloader, sampler_loader)
            logger.info(f"Final FID: {curr_score:.4f}")


# -----------------------------
# Main
# -----------------------------
def main(config_path: str = "config.yaml"):
    parser = argparse.ArgumentParser(description="Evaluate generative model")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--image_size", type=int, default=None, help="Model image size")
    parser.add_argument(
        "--image_channels", type=int, default=None, help="Model image channels"
    )
    args = parser.parse_args()
    config_path = args.config
    assert os.path.isfile(config_path), f"Config file {config_path} not found"

    # Load config
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup experiment
    run_name = f"{cfg['model']['type'].lower()}_{cfg['dataset']['type'].lower()}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = f"./eval_runs/{run_name}"
    logger.info(f"Saving intermidiate results to {run_dir}")
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(config_path, f"{run_dir}/config.yaml")

    # Setup model
    if args.image_size is not None and args.image_channels is not None:
        image_size = (args.image_size, args.image_size, args.image_channels)
        model = get_model(cfg["model"], dataloader=None, image_size=image_size)
    elif cfg["dataset"].get("type", None) is not None:
        dataloader = get_dataset(
            cfg["dataset"], cfg["training"].get("batch_size", None)
        )
        model = get_model(cfg["model"], dataloader=dataloader)
    else:
        raise ValueError(
            "Either dataset type or image_size and image_channels must be provided."
        )

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
