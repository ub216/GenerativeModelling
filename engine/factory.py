import inspect
import math
from typing import Any, Callable, Dict, List, Tuple

import torch
from loguru import logger

import helpers.custom_types as custom_types
import helpers.distributed_utils as dist_utils
import loaders
import losses
import metrics
import models
from engine.optimizer_manager import OptimizerManager


def _prepare_params(callable_: type | Callable, user_param_keys: set, all_params: dict) -> dict:
    """
    Validate user-provided config keys against a callable's signature and return a filtered
    params dict safe to unpack into callable_(**...).

    Works for both classes (inspects __init__) and plain functions.

    For callables without **kwargs (leaf models, loader functions):
      - Raises ValueError listing any user key absent from the signature.
      - Returns all_params filtered to only accepted keys, silently dropping
        factory-injected keys (e.g. image_size, in_channels, batch_size) the callable
        does not need.

    For callables with **kwargs (intermediate/wrapper models):
      - Returns all_params unchanged; leaf validation will surface unknown keys when
        forwarded kwargs reach a strict callable.
    """
    target = callable_.__init__ if isinstance(callable_, type) else callable_
    sig = inspect.signature(target)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return all_params

    accepted = {
        name
        for name, p in sig.parameters.items()
        if name != "self" and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    unknown = user_param_keys - accepted
    if unknown:
        raise ValueError(
            f"Unrecognised config parameters for {callable_.__name__}: {sorted(unknown)}. "
            f"Accepted: {sorted(accepted)}"
        )
    return {k: v for k, v in all_params.items() if k in accepted}


def get_dataset(cfg: Dict[str, Any], batch_size=None) -> torch.utils.data.DataLoader:
    name = cfg["type"].lower()
    params = cfg.get("params", {})
    # Capture keys the user explicitly set before factory injects batch_size.
    user_param_keys = set(params.keys())
    logger.info(f"Dataset params: {params}")
    if batch_size is not None and "batch_size" not in params:
        params["batch_size"] = batch_size
    elif "batch_size" in params and batch_size is not None and params["batch_size"] != batch_size:
        logger.warning(
            f"Warning: Overriding batch_size from {params['batch_size']} to {batch_size} "
            "as training batch size is provided."
        )
        params["batch_size"] = batch_size

    if name == "mnist":
        loader = loaders.get_mnist_dataloader(**_prepare_params(loaders.get_mnist_dataloader, user_param_keys, params))
    elif name == "celeb":
        loader = loaders.get_celeb_dataloader(**_prepare_params(loaders.get_celeb_dataloader, user_param_keys, params))
    elif name == "celeb_hq":
        loader = loaders.get_celeb_hq_dataloader(
            **_prepare_params(loaders.get_celeb_hq_dataloader, user_param_keys, params)
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # If distributed, wrap with DistributedSampler to split and shuffle across processes.
    # drop_last=True ensures all ranks see the same number of batches; without it the
    # sampler pads with repeated indices, causing some samples to be counted twice per epoch.
    if dist_utils.is_distributed():
        old_loader = loader
        # DataLoader does not expose a .shuffle attribute; detect from sampler type.
        # Distributed training loaders should always shuffle — non-shuffled loaders use SequentialSampler.
        shuffle = isinstance(old_loader.sampler, torch.utils.data.SequentialSampler) is False
        sampler = torch.utils.data.distributed.DistributedSampler(
            old_loader.dataset,
            shuffle=shuffle,
            drop_last=old_loader.drop_last,
        )
        loader = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=old_loader.batch_size,
            sampler=sampler,
            num_workers=old_loader.num_workers,
            pin_memory=old_loader.pin_memory,
        )
        # Copy custom attributes set by individual loaders (e.g. image_size)
        for attr in ("image_size",):
            if hasattr(old_loader, attr):
                setattr(loader, attr, getattr(old_loader, attr))
    return loader


def get_loss_function(cfg: Dict[str, Any]) -> torch.nn.Module:
    name = cfg["type"].lower()
    params = cfg.get("params", {})
    logger.info(f"Loss params: {params}")
    user_param_keys = set(params.keys())
    if name == "vae":
        return losses.VAELoss(**_prepare_params(losses.VAELoss, user_param_keys, params))
    elif name == "pair_mad":
        return losses.PairMADLoss(**_prepare_params(losses.PairMADLoss, user_param_keys, params))
    elif name == "pair_mse":
        return losses.PairMSELoss(**_prepare_params(losses.PairMSELoss, user_param_keys, params))
    elif name == "mean_flow_mse":
        return losses.MeanFlowMSELoss(**_prepare_params(losses.MeanFlowMSELoss, user_param_keys, params))
    elif name == "pair_smooth":
        return losses.PairSmoothLoss(**_prepare_params(losses.PairSmoothLoss, user_param_keys, params))
    elif name == "gan_hinge_loss":
        return losses.GANHingeLoss(**_prepare_params(losses.GANHingeLoss, user_param_keys, params))
    else:
        raise ValueError(f"Unknown loss function: {name}")


def get_model(
    cfg: Dict[str, Any],
    dataloader: torch.utils.data.DataLoader = None,
    image_size: Tuple[int, int, int] = None,
    build_ema: bool = True,
    ema_decay: float = 0.9999,
) -> Tuple[custom_types.GenBaseModel, custom_types.GenEMAModel | None]:
    if image_size is not None:
        h, w, c = image_size
    elif dataloader is not None:
        example_imgs, _ = next(iter(dataloader))
        _, c, h, w = example_imgs.shape
    else:
        raise ValueError("Either image_size or dataloader must be provided to infer image size.")
    name = cfg["type"].lower()
    params = cfg.get("params", {})
    # Capture keys the user explicitly set before factory injects image_size / in_channels.
    # _prepare_params uses this to distinguish user typos from factory-injected fields.
    user_param_keys = set(params.keys())

    # Compare image_size given in cfg and loaded from dataloader
    # We only consdier square images
    if params.get("image_size", None) is not None and params["image_size"] != h:
        logger.warning(f"Overriding model image_size to {h} similar to the input recived from Dataloader")
    params["image_size"] = h

    # Compare in_channels given in cfg and loaded from dataloader
    if params.get("in_channels", None) is not None and params["in_channels"] != c:
        logger.warning(f"Overriding model in_channels to {c} similar to the input recived from Dataloader")
    params["in_channels"] = c

    logger.info(f"Model params: {params}")
    if name == "vae":
        if h != w:
            raise ValueError(f"VAE implementation requires square images, got {h}x{w}")
        vae_params = _prepare_params(models.VAE, user_param_keys, params)
        return models.VAE(**vae_params), (
            models.EMAModel(models.VAE(**vae_params), decay=ema_decay) if build_ema else None
        )
    if name == "gan":
        if h != w:
            raise ValueError(f"GAN implementation requires square images, got {h}x{w}")
        gan_params = _prepare_params(models.GAN, user_param_keys, params)
        return models.GAN(**gan_params), (
            models.EMAModel(models.GAN(**gan_params), decay=ema_decay) if build_ema else None
        )
    elif name == "diffusion":
        diffusion_params = _prepare_params(models.DiffusionModel, user_param_keys, params)
        return models.DiffusionModel(**diffusion_params), (
            models.EMAModel(models.DiffusionModel(**diffusion_params), decay=ema_decay) if build_ema else None
        )
    elif name == "flow":
        flow_params = _prepare_params(models.FlowModel, user_param_keys, params)
        return models.FlowModel(**flow_params), (
            models.EMAModel(models.FlowModel(**flow_params), decay=ema_decay) if build_ema else None
        )
    elif name == "latent_diffusion":
        latent_diffusion_params = _prepare_params(models.LatentDiffusionModel, user_param_keys, params)
        return models.LatentDiffusionModel(**latent_diffusion_params), (
            models.EMAModel(models.LatentDiffusionModel(**latent_diffusion_params), decay=ema_decay)
            if build_ema
            else None
        )
    elif name == "dpo_latent_diffusion":
        dpo_params = _prepare_params(models.DPOLatentDiffusionModel, user_param_keys, params)
        return models.DPOLatentDiffusionModel(**dpo_params), (
            models.EMAModel(models.DPOLatentDiffusionModel(**dpo_params), decay=ema_decay) if build_ema else None
        )
    elif name == "latent_flow":
        latent_flow_params = _prepare_params(models.LatentFlowModel, user_param_keys, params)
        return models.LatentFlowModel(**latent_flow_params), (
            models.EMAModel(models.LatentFlowModel(**latent_flow_params), decay=ema_decay) if build_ema else None
        )
    elif name == "latent_mean_flow":
        latent_mean_flow_params = _prepare_params(models.LatentMeanFlowModel, user_param_keys, params)
        return models.LatentMeanFlowModel(**latent_mean_flow_params), (
            models.EMAModel(models.LatentMeanFlowModel(**latent_mean_flow_params), decay=ema_decay)
            if build_ema
            else None
        )
    elif name == "mean_flow":
        mean_flow_params = _prepare_params(models.MeanFlowModel, user_param_keys, params)
        return models.MeanFlowModel(**mean_flow_params), (
            models.EMAModel(models.MeanFlowModel(**mean_flow_params), decay=ema_decay) if build_ema else None
        )
    else:
        raise ValueError(f"Unknown model type: {name}")


def get_metrics(cfg: Dict[str, Any]) -> List[torch.nn.Module]:
    metric = []
    if cfg is None:
        return metric
    for name, key in cfg.items():
        name = name.lower()
        params = key.get("params", {})
        user_param_keys = set(params.keys())

        logger.info(f"Metric params: {params}")
        if name == "fid":
            metric.append(metrics.FIDInception(**_prepare_params(metrics.FIDInception, user_param_keys, params)))
        elif name == "cmmd":
            metric.append(metrics.CMMDClip(**_prepare_params(metrics.CMMDClip, user_param_keys, params)))
        else:
            raise ValueError(f"Unknown metric: {name}")
    return metric


def _get_optimizer(name: str, paramertes, options: Dict[str, Any]) -> torch.optim.Optimizer:
    if name.lower() == "adam":
        return torch.optim.Adam(params=paramertes, **options)
    elif name.lower() == "adamw":
        return torch.optim.AdamW(params=paramertes, **options)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    params: Dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler:
    base_lr = optimizer.param_groups[0]["lr"]
    if name.lower() == "cosine_warmup":
        warmup_steps = params.get("warmup_steps", 0)
        eta_min = params.get("eta_min", 0.0)
        decay_steps = max(1, total_steps - warmup_steps)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale cosine output from [0, 1] down to [eta_min/base_lr, 1].
            return eta_min / base_lr + (1.0 - eta_min / base_lr) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif name.lower() == "cosine":
        eta_min = params.get("eta_min", 0.0)
        decay_steps = max(1, total_steps)

        def lr_lambda(step: int) -> float:
            progress = step / decay_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_min / base_lr + (1.0 - eta_min / base_lr) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler: {name!r}. Available: cosine, cosine_warmup")


def get_optimizer_manager(
    cfg: Dict[str, Any],
    model: custom_types.GenBaseModel,
    amp_dtype: torch.dtype | None = torch.float16,
    total_steps: int = 0,
) -> OptimizerManager:
    optimizer = {}
    if "type" in cfg:
        # single optimizer for all parameters
        optimizer["all"] = _get_optimizer(cfg["type"], model.parameters(), cfg.get("params", {}))
    else:
        for name, key in cfg.items():
            module = getattr(model, name, None)
            if module is None:
                raise ValueError(f"'{name}' does not exist in model parameter groups")
            optimizer[name] = _get_optimizer(key["type"], module.parameters(), key.get("params", {}))

    scheduler_cfg = cfg.get("scheduler")
    schedulers = {}
    if scheduler_cfg is not None:
        if total_steps == 0:
            logger.warning(
                "Scheduler configured but total_steps=0 — LR will not decay. "
                "Pass total_steps to get_optimizer_manager."
            )
        for key, opt in optimizer.items():
            schedulers[key] = _get_scheduler(scheduler_cfg["type"], opt, total_steps, scheduler_cfg.get("params", {}))

    # GradScaler is only needed for FP16 overflow recovery; BF16 and disabled AMP don't need it
    use_scaler = amp_dtype == torch.float16
    optimizer_manager = OptimizerManager(
        optimizer,
        model=model,
        use_scaler=use_scaler,
        accumulate_steps=cfg.get("accumulate_steps", 1),
        max_grad_norm=cfg.get("max_grad_norm", float("inf")),
        schedulers=schedulers,
    )
    return optimizer_manager
