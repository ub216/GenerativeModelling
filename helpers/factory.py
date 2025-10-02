from typing import Any, Dict, List

import torch
from loguru import logger

import helpers.custom_types as custom_types
import loaders
import losses
import metrics
import models


def get_dataset(cfg: Dict[str, Any], batch_size=None) -> torch.utils.data.DataLoader:
    name = cfg["type"].lower()
    params = cfg.get("params", {})
    logger.info(f"Dataset params: {params}")
    if batch_size is not None and "batch_size" not in params:
        params["batch_size"] = batch_size
    elif (
        "batch_size" in params
        and batch_size is not None
        and params["batch_size"] != batch_size
    ):
        logger.warning(
            f"Warning: Overriding batch_size from {params['batch_size']} to {batch_size} as training batch size is provided."
        )
        params["batch_size"] = batch_size

    if name == "mnist":
        return loaders.get_mnist_dataloader(**params)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_loss_function(cfg: Dict[str, Any]) -> torch.nn.Module:
    name = cfg["type"].lower()
    params = cfg.get("params", {})
    logger.info(f"Loss params: {params}")
    if name == "vae":
        return losses.VAELoss(**params)
    elif name == "pair_mse":
        return losses.PairMSELoss(**params)
    else:
        raise ValueError(f"Unknown loss function: {name}")


def get_model(
    cfg: Dict[str, Any], dataloader: torch.utils.data.DataLoader
) -> custom_types.GenBaseModel:
    example_imgs, _ = next(iter(dataloader))
    _, c, h, w = example_imgs.shape
    name = cfg["type"].lower()
    params = cfg.get("params", {})

    # Compare image_size given in cfg and loaded from dataloader
    # We only consdier square images
    if params.get("image_size", None) is not None and params["image_size"] != h:
        logger.warning(
            f"Overriding model image_size to {h} similar to the input recived from Dataloader"
        )
    params["image_size"] = h

    # Compare in_channels given in cfg and loaded from dataloader
    if params.get("in_channels", None) is not None and params["in_channels"] != c:
        logger.warning(
            f"Overriding model in_channels to {c} similar to the input recived from Dataloader"
        )
    params["in_channels"] = c

    logger.info(f"Model params: {params}")
    if name == "vae":
        assert h == w, f"VAE implementation can only handle square images for now"
        return models.VAE(**params)
    elif name == "diffusion":
        return models.DiffusionModel(**params)
    elif name == "flow":
        return models.FlowModel(**params)
    else:
        raise ValueError(f"Unknown model type: {name}")


def get_metrics(cfg: Dict[str, Any]) -> List[torch.nn.Module]:
    metric = []
    if cfg is None:
        return metric
    for name, key in cfg.items():
        name = name.lower()
        params = key.get("params", {})

        logger.info(f"Metric params: {params}")
        if name == "fid":
            metric.append(metrics.FIDInception(**params))
        else:
            raise ValueError(f"Unknown metric: {name}")
    return metric
