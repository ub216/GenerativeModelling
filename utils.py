import random

import cv2
import numpy as np


def save_eval_results(samples, filename="generated_samples.png", conditioning=None):
    """
    Save generated samples in a grid format with optional text overlay.
    if conditioning is provided, it is overlayed on the corresponding input to the top-left corner.
    if unconditioned then nothing is overlayed.
    """
    assert len(samples.shape) == 4  # (N, C, H, W)
    num_samples = samples.size(0)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    sample_height, sample_width = samples.size(2), samples.size(3)

    # grayscale grid canvas
    grid_image = np.zeros(
        (grid_size * sample_height, grid_size * sample_width), dtype=np.uint8
    )
    # normalize conditioning list length if provided
    if conditioning is not None:
        assert (
            len(conditioning) == num_samples
        ), f"conditioning length {len(conditioning)} must match samples {num_samples}"

    for idx in range(num_samples):
        row = idx // grid_size
        col = idx % grid_size
        img = samples[idx].squeeze().cpu().numpy() * 255
        img = img.astype(np.uint8)

        # overlay text if conditioning provided
        if conditioning is not None and conditioning[idx] != "":
            label = str(conditioning[idx])
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


def drop_condition(conditioning, r):
    """
    conditioning: Conditioning text
    r: percentage (0–1) of elements to replace with null condition ""
    """
    assert r < 1
    N = len(conditioning)
    k = int(N * r)  # how many to blank out
    indices = random.sample(range(N), k)  # pick k random positions

    # copy list so original is not modified
    drop_conditioning = conditioning[:]
    for i in indices:
        drop_conditioning[i] = ""
    return drop_conditioning
