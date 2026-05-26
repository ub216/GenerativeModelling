from typing import List, Tuple

import torch
from loguru import logger

import helpers.custom_types as custom_types
import metrics
from helpers.diffusion_utils import drop_condition
from helpers.utils import save_eval_results


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
            samples = model.sample(num_samples, device, image_size, batch_size=num_samples)

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
    scores = []
    for metric in compute_metrics:
        if isinstance(metric, metrics.ImageDistributionMetric):
            sampler_loader = model.wrap_sampler_to_loader(
                num_samples=metric.samples,
                device=device,
                image_size=dataloader.image_size,
                batch_size=dataloader.batch_size,
            )
            score = metric(dataloader, sampler_loader)
            scores.append(f"{metric.name}: {score:.4f}")
    logger.info(f"Final Metrics: {', '.join(scores)}")
