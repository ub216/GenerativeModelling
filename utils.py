import cv2
import numpy as np


def save_eval_results(samples, filename="generated_samples.png"):
    """Save generated samples in a grid format."""
    assert len(samples.shape) == 4  # (N, C, H, W)
    num_samples = samples.size(0)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    sample_height, sample_width = samples.size(2), samples.size(3)

    grid_image = np.zeros(
        (grid_size * sample_height, grid_size * sample_width), dtype=np.uint8
    )

    for idx in range(num_samples):
        row = idx // grid_size
        col = idx % grid_size
        img = samples[idx].squeeze().cpu().numpy() * 255
        img = img.astype(np.uint8)
        grid_image[
            row * sample_height : (row + 1) * sample_height,
            col * sample_width : (col + 1) * sample_width,
        ] = img

    cv2.imwrite(filename, grid_image)
