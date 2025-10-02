import random

import cv2
import numpy as np


def save_eval_results(samples, filename="generated_samples.png", cond=None):
    """Save generated samples in a grid format with optional text overlay."""
    assert len(samples.shape) == 4  # (N, C, H, W)
    num_samples = samples.size(0)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    sample_height, sample_width = samples.size(2), samples.size(3)

    # grayscale grid canvas
    grid_image = np.zeros(
        (grid_size * sample_height, grid_size * sample_width), dtype=np.uint8
    )
    # normalize cond list length if provided
    if cond is not None:
        assert (
            len(cond) == num_samples
        ), f"cond length {len(cond)} must match samples {num_samples}"

    for idx in range(num_samples):
        row = idx // grid_size
        col = idx % grid_size
        img = samples[idx].squeeze().cpu().numpy() * 255
        img = img.astype(np.uint8)

        # overlay text if cond provided
        if cond is not None and cond[idx] != "":
            label = str(cond[idx])
            # draw text directly on the image
            img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                img_colored,
                label,
                (2, 6),  # top-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.2,  # font scale
                (255, 255, 255),  # white text
                1,
                cv2.LINE_AA,
            )
            img = cv2.cvtColor(img_colored, cv2.COLOR_BGR2GRAY)

        grid_image[
            row * sample_height : (row + 1) * sample_height,
            col * sample_width : (col + 1) * sample_width,
        ] = img

    cv2.imwrite(filename, grid_image)


def drop_condition(cond, r):
    """
    cond: Conditioning text
    r: percentage (0–1) of elements to replace with null condition ""
    """
    assert r < 1
    N = len(cond)
    k = int(N * r)  # how many to blank out
    indices = random.sample(range(N), k)  # pick k random positions

    # copy list so original is not modified
    drop_cond = cond[:]
    for i in indices:
        drop_cond[i] = ""
    return drop_cond
