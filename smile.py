import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from edit_images.ddim_invert_edit import (
    ddim_edit_from_noise,
    ddim_invert,
    linear_cfg_ramp,
    make_ddim_time_pairs,
)
from edit_images.face_align import (
    FaceAligner,
    build_target_landmark_template_from_aligned_images,
)
from helpers.factory import get_model
from models.base_model import BaseModel


def preprocess_for_model(
    rgb64: np.ndarray, renormalise: bool, device: str
) -> torch.Tensor:
    x = torch.from_numpy(rgb64).float() / 255.0  # (H,W,C)
    x = x.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)
    if renormalise:
        x = x * 2.0 - 1.0
    return x.to(device)


def postprocess_from_model(x: torch.Tensor, renormalise: bool) -> np.ndarray:
    x = x.detach().cpu()
    if renormalise:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return (x * 255).astype(np.uint8)


@torch.no_grad()
def edit_smile(
    model: BaseModel,
    aligner: FaceAligner,
    input_bgr_path: str,
    prompt: str,
    T_test: int = 50,
    step_percent: float = 0.4,
) -> np.ndarray:
    bgr = cv2.imread(input_bgr_path)

    # detect face, align
    aligned_rgb, meta = aligner.align_largest(bgr)
    if aligned_rgb is None:
        raise RuntimeError("No face detected")

    # invert with empty conditioning (reconstruct prior)
    x0 = preprocess_for_model(aligned_rgb, renormalise=model.renormalise, device="cuda")
    inv_cond = [""] if model.has_conditional_generation else None
    full_inc, full_dec = make_ddim_time_pairs(model.timesteps, T_test, device="cuda")

    # 2. Slice them to only include the first 40% of the steps
    # If T_test is 50, 40% of the steps is 20 steps.
    num_steps = int(len(full_inc) * step_percent)

    inc_pairs = full_inc[:num_steps]
    dec_pairs = full_dec[-num_steps:]
    xT = ddim_invert(model, x0, inv_cond, inc_pairs=inc_pairs, device="cuda")

    # edit with smile conditioning + CFG ramp
    edit_cond = [prompt] if model.has_conditional_generation else None
    cfg_sched = linear_cfg_ramp(
        cfg_start=1.5, cfg_end=model.sample_condition_weight
    )  # tweak
    x_edit = ddim_edit_from_noise(
        model, xT, edit_cond, dec_pairs=dec_pairs, device="cuda", cfg_schedule=cfg_sched
    )

    edited_aligned_rgb = postprocess_from_model(x_edit, renormalise=model.renormalise)
    out_bgr = aligner.unalign_and_paste(edited_aligned_rgb, bgr, meta, feather=10)
    return out_bgr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diffusion_celeb_smiling.yaml",
        help="Model config path",
    )
    parser.add_argument("--image_size", type=int, default=64, help="Model image size")
    parser.add_argument("--prompt", type=str, default="smiling", help="Editing prompt")
    parser.add_argument(
        "--T_test",
        type=int,
        default=5,
        help="Number of DDIM steps for inversion/editing (5-50). Higher = larger smile but potentially less identity preservation.",
    )
    parser.add_argument(
        "--step_percent",
        type=float,
        default=0.4,
        help="Percentage of steps to use for inversion/editing. We don't need to go to full noise.",
    )
    args = parser.parse_args()

    assert os.path.isfile(args.config), f"Config file {args.config} not found"
    assert os.path.isfile(args.input), f"Input image file {args.input} not found"
    filename, ext = os.path.splitext(os.path.basename(args.input))
    os.makedirs("outputs_smiling", exist_ok=True)
    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup model
    model = get_model(
        cfg["model"], None, image_size=(args.image_size, args.image_size, 3)
    )

    # Load weights with preference for EMA weights

    if cfg["model"].get("checkpoint", None) is not None:
        ckpt = cfg["model"]["checkpoint"]
        checkpoint = torch.load(ckpt, map_location="cpu")
        if "model_ema_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_ema_state_dict"])
            print(f"Loaded EMA weights from {ckpt}")
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded model weights from {ckpt}")
        else:
            raise RuntimeError("Checkpoint does not contain model weights.")
    else:
        raise RuntimeError("Please provide a trained model checkpoint for editing.")

    model.eval()
    model = model.to("cuda")

    aligner = FaceAligner(image_size=args.image_size, device="cuda")

    out = edit_smile(
        model,
        aligner,
        args.input,
        prompt=args.prompt,
        T_test=args.T_test,
        step_percent=args.step_percent,
    )
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB))
    plt.title("Input")
    plt.subplot(1, 2, 2)
    plt.title("Edited")
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.savefig(
        f"outputs_smiling/comparison_{filename}.png", dpi=200, bbox_inches="tight"
    )  # saves the whole figure
    plt.show()
