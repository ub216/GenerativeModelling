from functools import lru_cache
from typing import List, Optional

import cv2
import numpy as np
import torch
from loguru import logger


@lru_cache(None)
def log_once_warning(msg):
    logger.warning(msg)


@lru_cache(None)
def log_once_info(msg):
    logger.info(msg)


def save_eval_results(
    samples: torch.Tensor,
    filename: str = "generated_samples.png",
    conditioning: Optional[List[str]] = None,
):
    """
    Save generated samples in a grid format with optional text overlay.
    if conditioning is provided, it is overlayed on the corresponding input to the top-left corner.
    if unconditioned then nothing is overlayed.
    """
    assert len(samples.shape) == 4  # (N, C, H, W)
    num_samples = samples.size(0)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    sample_height, sample_width, sample_channels = (
        samples.size(2),
        samples.size(3),
        samples.size(1),
    )

    # grayscale grid canvas
    grid_image = np.zeros(
        (grid_size * sample_height, grid_size * sample_width, sample_channels),
        dtype=np.uint8,
    )
    # normalize conditioning list length if provided
    if conditioning is not None:
        assert (
            len(conditioning) == num_samples
        ), f"conditioning length {len(conditioning)} must match samples {num_samples}"

    for idx in range(num_samples):
        row = idx // grid_size
        col = idx % grid_size
        img = samples[idx].permute(1, 2, 0).cpu().numpy() * 255
        img = img.astype(np.uint8)

        # overlay text if conditioning provided
        if conditioning is not None and conditioning[idx] != "":
            label = str(conditioning[idx])
            # draw text directly on the image
            img_colored = img
            if sample_channels == 1:
                img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img_colored = cv2.putText(
                np.ascontiguousarray(img_colored),
                label,
                (2, 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.2,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            img = img_colored
            if sample_channels == 1:
                img = cv2.cvtColor(img_colored, cv2.COLOR_BGR2GRAY)
        grid_image[
            row * sample_height : (row + 1) * sample_height,
            col * sample_width : (col + 1) * sample_width,
        ] = img

    # convert colour format from RGB to BGR for OpenCV
    # remove extra channel dimension if grayscale
    grid_image = grid_image[:, :, ::-1].squeeze()
    cv2.imwrite(filename, grid_image)
