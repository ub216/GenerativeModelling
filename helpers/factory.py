import inspect
from typing import Any, Dict, List, Tuple

import torch
from loguru import logger

import helpers.custom_types as custom_types
import loaders
import losses
import metrics
import models
from helpers.optimizer_manager import OptimizerManager


def _prepare_params(cls: type, user_param_keys: set, all_params: dict) -> dict:
    """
    Validate user-provided config keys against cls.__init__ and return a filtered
    params dict safe to unpack into cls(**...).

    For constructors without **kwargs (leaf models):
      - Raises ValueError listing any user key absent from the constructor's signature.
      - Returns all_params filtered to only accepted keys, silently dropping
        factory-injected keys (e.g. image_size, in_channels) the constructor does not need.

    For constructors with **kwargs (intermediate/wrapper models):
      - Returns all_params unchanged; leaf constructor validation will surface unknown
        keys when forwarded kwargs reach a strict leaf.
    """
    sig = inspect.signature(cls.__init__)
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
            f"Unrecognised config parameters for {cls.__name__}: {sorted(unknown)}. " f"Accepted: {sorted(accepted)}"
        )
    return {k: v for k, v in all_params.items() if k in accepted}


def get_dataset(cfg: Dict[str, Any], batch_size=None) -> torch.utils.data.DataLoader:
    name = cfg["type"].lower()
    params = cfg.get("params", {})
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
        return loaders.get_mnist_dataloader(**params)
    elif name == "celeb":
        return loaders.get_celeb_dataloader(**params)
    elif name == "celeb_hq":
        return loaders.get_celeb_hq_dataloader(**params)
    else:
        raise ValueError(f"Unknown dataset: {name}")


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
        assert h == w, "VAE implementation can only handle square images for now"
        vae_params = _prepare_params(models.VAE, user_param_keys, params)
        return models.VAE(**vae_params), (
            models.EMAModel(models.VAE(**vae_params), decay=ema_decay) if build_ema else None
        )
    if name == "gan":
        assert h == w, "GAN implementation can only handle square images for now"
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


def get_optimizer_manager(
    cfg: Dict[str, Any],
    model: custom_types.GenBaseModel,
    amp_dtype: torch.dtype | None = torch.float16,
) -> OptimizerManager:
    optimizer = {}
    if "type" in cfg:
        # single optimizer for all parameters
        optimizer["all"] = _get_optimizer(cfg["type"], model.parameters(), cfg.get("params", {}))
    else:
        for name, key in cfg.items():
            module = getattr(model, name, None)
            assert module is not None, f"{module} does not exists in model parameter groups"
            optimizer[name] = _get_optimizer(key["type"], module.parameters(), key.get("params", {}))
    # GradScaler is only needed for FP16 overflow recovery; BF16 and disabled AMP don't need it
    use_scaler = amp_dtype == torch.float16
    optimizer_manager = OptimizerManager(
        optimizer,
        model=model,
        use_scaler=use_scaler,
        accumulate_steps=cfg.get("accumulate_steps", 1),
        max_grad_norm=cfg.get("max_grad_norm", float("inf")),
    )
    return optimizer_manager
