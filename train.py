import argparse
import os
import shutil
import time

import torch
import yaml
from loguru import logger

import helpers.distributed_utils as dist_utils
import wandb
from engine.factory import get_dataset, get_loss_function, get_metrics, get_model, get_optimizer_manager
from engine.trainer import train
from helpers.utils import cuda_available


# -----------------------------
# Main
# -----------------------------
def main(config_path: str = "config.yaml"):
    # Initialize distributed training (if applicable)
    dist_utils.init_distributed_training()
    try:
        parser = argparse.ArgumentParser(description="Train generative model")
        parser.add_argument("--config", type=str, required=True, help="Config file path")
        args = parser.parse_args()
        config_path = args.config
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found")

        # Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        # Each distributed process must own its own GPU; fall back to config for single-process
        if dist_utils.is_distributed() and cuda_available():
            device = f"cuda:{os.environ['LOCAL_RANK']}"
        else:
            device = cfg["training"]["device"]

        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        # Resolve AMP dtype from config; "auto" picks BF16 on supported hardware
        _dtype_cfg = cfg["training"].get("amp_dtype", "auto")
        if _dtype_cfg == "auto":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif _dtype_cfg == "bfloat16":
            amp_dtype = torch.bfloat16
        elif _dtype_cfg == "float16":
            amp_dtype = torch.float16
        else:
            amp_dtype = None  # AMP disabled
        logger.info(f"AMP dtype: {amp_dtype} (config: {_dtype_cfg!r})")

        # Setup experiment
        torch.manual_seed(cfg["experiment"].get("seed", 42))
        logger.info(f"Torch seed set to {cfg['experiment'].get('seed', 42)}")
        run_name = f"{cfg['model']['type'].lower()}_{cfg['dataset']['type'].lower()}_{time.strftime('%Y%m%d-%H%M%S')}"
        if dist_utils.is_main_process():
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
        # Also initialise EMA model for stable sampling
        model, model_ema = get_model(cfg["model"], dataloader, ema_decay=cfg["training"].get("ema_decay", 0))

        model.to(device)
        model_ema.to(device)

        if dist_utils.is_distributed():
            # EMA is not wrapped in DDP: it never runs backward, so no gradient
            # sync is needed. Shadow weights stay in sync naturally — DDP keeps
            # online params identical on all ranks, and the EMA update is
            # deterministic, so shadow params remain identical too.
            local_rank = int(os.environ["LOCAL_RANK"])
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank] if cuda_available() else None,
            )

        if cfg["training"].get("compile", False):
            logger.info("Compiling model with torch.compile(mode='reduce-overhead')...")
            model = torch.compile(model, mode="reduce-overhead")
            model_ema.model = torch.compile(model_ema.model, mode="reduce-overhead")

        # Setup loss
        criterion = get_loss_function(cfg["loss"])

        # Setup Optimizer. This is done after the model is intialized
        optimizer_manager = get_optimizer_manager(cfg["optimizer"], model, amp_dtype=amp_dtype)

        # Load checkpoint if provided
        start_epoch = 0
        if cfg["model"].get("checkpoint", None) is not None:
            ckpt = cfg["model"]["checkpoint"]
            # weights_only=True: load model/EMA weights but skip optimizer state and epoch
            # counter so stage-2 fine-tuning starts with a fresh optimizer and LR schedule
            weights_only = cfg["model"].get("checkpoint_weights_only", False)
            checkpoint = torch.load(ckpt, map_location=device)
            dist_utils.unwrap_model(model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            dist_utils.unwrap_model(model_ema).load_state_dict(checkpoint["model_ema_state_dict"], strict=False)
            if not weights_only:
                if "optimizer_state_dict" in checkpoint:
                    optimizer_manager.load_state_dict(checkpoint["optimizer_state_dict"])
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Loaded checkpoint from {ckpt} (weights_only={weights_only})")

        # Metrics
        compute_metrics = get_metrics(cfg.get("metrics", None))
        metric_interval = cfg["training"].get("metric_interval", None)
        if compute_metrics and metric_interval is None:
            logger.warning(
                "Metrics are defined but 'metric_interval' is not set in training config. "
                "Metric computation will be skipped. Set training.metric_interval to enable it."
            )

        # Training loop
        train(
            model,
            model_ema,
            dataloader,
            optimizer_manager,
            criterion,
            compute_metrics,
            metric_interval=metric_interval,
            sample_interval=cfg["training"].get("sample_interval", 1),
            device=device,
            epochs=cfg["training"]["epochs"],
            start_epoch=start_epoch,
            save_dir=f"./runs/{run_name}",
            save_after_epoch=cfg["training"].get("save_after_epoch", float("inf")),
            amp_dtype=amp_dtype,
            num_fixed_samples=cfg["training"].get("num_fixed_samples", 0),
            fixed_seed=cfg["experiment"].get("seed", 42),
        )
    finally:
        dist_utils.cleanup_distributed_training()


if __name__ == "__main__":
    main()
