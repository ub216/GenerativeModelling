import random
from typing import List, Optional

import cv2
import numpy as np
import torch


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


def drop_condition(conditioning: List[str], r: float) -> List[str]:
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


def update_step(
    losses,
    optimizers,
    scalers=None,
    step: int = 1,
    accumulation: int = 1,
):
    """
    Update step with optional AMP scaling and gradient accumulation.

    Args:
        losses (dict[str, torch.Tensor]): dict of losses, e.g. {"gen": g_loss, "disc": d_loss}
        optimizers (dict[str, torch.optim.Optimizer]): dict of optimizers
        scalers (dict[str, torch.cuda.amp.GradScaler] | None): dict of scalers, same keys as optimizers
        step (int): current global step
        accumulation (int): number of steps to accumulate gradients before optimizer.step()
    """
    assert set(losses.keys()) == set(
        optimizers.keys()
    ), f"Loss keys {losses.keys()} must match optimizer keys {optimizers.keys()}"

    if scalers is not None:
        assert set(scalers.keys()) == set(
            optimizers.keys()
        ), f"Scaler keys {scalers.keys()} must match optimizer keys {optimizers.keys()}"

    # scale losses by accumulation to keep effective lr same
    scaled_losses = {k: v.mean() / accumulation for k, v in losses.items()}

    # backward pass
    for key, opt in optimizers.items():
        if scalers is not None:
            scalers[key].scale(scaled_losses[key]).backward()
        else:
            scaled_losses[key].backward()

    # only step/update every `accumulation` steps
    if step % accumulation == 0:
        for key, opt in optimizers.items():
            if scalers is not None:
                scalers[key].step(opt)
                scalers[key].update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
