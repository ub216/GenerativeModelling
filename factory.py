from loguru import logger

import loaders
import losses
import metrics
import models


def get_dataset(cfg, batch_size=None):
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


def get_loss_function(cfg):
    name = cfg["type"].lower()
    params = cfg.get("params", {})
    logger.info(f"Loss params: {params}")
    if name == "vae":
        return losses.VAELoss(**params)
    elif name == "pair_mse":
        return losses.PairMSELoss(**params)
    else:
        raise ValueError(f"Unknown loss function: {name}")


def get_model(cfg):
    name = cfg["type"].lower()
    params = cfg.get("params", {})
    logger.info(f"Model params: {params}")
    if name == "vae":
        return models.VAE(**params)
    elif name == "diffusion":
        return models.DiffusionModel(**params)
    else:
        raise ValueError(f"Unknown model type: {name}")


def get_metrics(cfg):
    metric = []
    for name, key in cfg.items():
        name = name.lower()
        params = key.get("params", {})

        logger.info(f"Metric params: {params}")
        if name == "fid":
            metric.append(metrics.FIDInception(**params))
        else:
            raise ValueError(f"Unknown metric: {name}")
    return metric
