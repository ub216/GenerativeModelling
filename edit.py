import argparse
import os
from typing import List, Optional, Tuple

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from edit_images.ddim_edit import (
    ddim_edit_from_noise,
    ddim_invert,
    linear_cfg_ramp,
    make_ddim_time_pairs,
    sdedit_add_noise,
)
from edit_images.face_align import FaceAligner
from edit_images.face_verifier import FaceVerifier
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


def edit_smile(
    model: BaseModel,
    aligner: FaceAligner,
    verifier: FaceVerifier,
    input_bgr_path: str,
    prompt: str,
    T_test: int = 50,
    step_percent: float = 0.4,
    id_scale: float = 0.0,
    edit_mode: str = "invert",
) -> Tuple[np.ndarray, float]:
    bgr = cv2.imread(input_bgr_path)

    # detect face, align
    aligned_rgb, meta = aligner.align_largest(bgr)
    if aligned_rgb is None:
        raise RuntimeError("No face detected")

    # invert with empty conditioning (reconstruct prior)
    x0 = preprocess_for_model(aligned_rgb, renormalise=model.renormalise, device="cuda")
    inv_cond = [""] if model.has_conditional_generation else None
    full_inc, full_dec = make_ddim_time_pairs(model.timesteps, T_test, device="cuda")

    # slice them to only include the first 40% of the steps
    num_steps = int(len(full_inc) * step_percent)

    if edit_mode == "invert":
        # Inversion: Deterministic mapping to noise
        inv_cond = [""] if model.has_conditional_generation else None
        inc_pairs = full_inc[:num_steps]
        xT = ddim_invert(model, x0, inv_cond, inc_pairs=inc_pairs, device="cuda")
    else:
        # SDEdit: Stochastic addition of noise
        # The starting timestep index for denoising
        t_start_idx = full_inc[num_steps - 1][1]
        xT = sdedit_add_noise(x0, t_start_idx, model, device="cuda")

    # edit with smile conditioning + CFG ramp
    dec_pairs = full_dec[-num_steps:]
    edit_cond = [prompt] if model.has_conditional_generation else None
    cfg_sched = linear_cfg_ramp(cfg_start=1.5, cfg_end=model.sample_condition_weight)

    # identity guidance function if scale > 0.
    guidance_fn = None
    if id_scale > 0:
        # Get target embedding from original image (x0)
        with torch.no_grad():
            if not model.renormalise:
                x0_for_id = x0 * 2.0 - 1.0  # Map [0,1] to [-1,1]
            else:
                x0_for_id = x0
            target_emb = verifier.get_embedding_differentiable(x0_for_id).detach()

        def _id_guidance(x_in, x0_pred, step_idx, total_steps):
            # The code calculates a ramp weight based on progress.
            # Early steps (high noise) get weight ~0, later steps get weight ~1.
            ramp = step_idx / max(1, total_steps - 1)
            current_scale = id_scale * ramp

            # We want to maximize Cosine Similarity, so we minimize (1 - CosSim)
            curr_emb = verifier.get_embedding_differentiable(x0_pred)
            sim = (curr_emb * target_emb).sum()
            loss = (1 - sim) * current_scale
            return loss

        guidance_fn = _id_guidance

    x_edit = ddim_edit_from_noise(
        model,
        xT,
        edit_cond,
        dec_pairs=dec_pairs,
        device="cuda",
        cfg_schedule=cfg_sched,
        guidance_fn=guidance_fn,
    )

    edited_aligned_rgb = postprocess_from_model(x_edit, renormalise=model.renormalise)
    out_bgr = aligner.unalign_and_paste(edited_aligned_rgb, bgr, meta, feather=10)
    similarity = verifier.get_similarity(aligned_rgb, edited_aligned_rgb)
    return out_bgr, similarity


def make_progressive_video(
    model: BaseModel,
    aligner: FaceAligner,
    verifier: FaceVerifier,
    input_bgr_path: str,
    prompt: str,
    output_path: str,
    T_test: int = 50,
    step_percent: float = 0.4,
    id_scale: float = 20.0,
    num_frames: int = 15,
    temp_scale: float = 5.0,  # Strength of temporal consistency guidance
):
    """
    Generate a video transitioning from the original face to the smiling face.
    Use both Identity Guidance (to keep the person recognizable) and
    Temporal Guidance (to ensure the transition is smooth).
    """
    bgr = cv2.imread(input_bgr_path)
    aligned_rgb, meta = aligner.align_largest(bgr)
    if aligned_rgb is None:
        raise RuntimeError("No face detected")

    x0 = preprocess_for_model(aligned_rgb, renormalise=model.renormalise, device="cuda")
    inv_cond = [""] if model.has_conditional_generation else None
    full_inc, full_dec = make_ddim_time_pairs(model.timesteps, T_test, device="cuda")

    # Slice steps
    steps_count = int(len(full_inc) * step_percent)
    inc_pairs = full_inc[:steps_count]
    dec_pairs = full_dec[-steps_count:]

    # 1. Invert ONCE to get the starting noise
    print("Inverting...")
    xT_base = ddim_invert(model, x0, inv_cond, inc_pairs=inc_pairs, device="cuda")

    # 2. Get Reference Embedding (Identity)
    with torch.no_grad():
        id_emb = verifier.get_embedding_differentiable(x0).detach()

    frames = []
    # track the previous frame's latent/prediction for temporal consistency
    prev_x0_pred = None

    print(f"Generating {num_frames} frames...")

    for i in range(num_frames):
        # ramp up the CFG scale to increase the smile intensity
        progress = i / (num_frames - 1)
        # CFG starts at 0.0 (unconditioned/reconstruction) and goes to model.weight
        target_cfg = model.sample_condition_weight * progress

        # Schedule: Start gentle, end strong
        cfg_sched = linear_cfg_ramp(cfg_start=0.0, cfg_end=target_cfg)

        # Define Guidance for this frame
        def _video_guidance(x_in, x0_pred, step_idx, total_steps):
            loss = torch.tensor(0.0, device="cuda")

            # The code applies ramping to the identity loss here as well.
            # It ensures guidance is stronger when the image is cleaner.
            ramp = step_idx / max(1, total_steps - 1)
            current_id_scale = id_scale * ramp

            # A. Identity Loss
            if current_id_scale > 0:
                curr_emb = verifier.get_embedding_differentiable(x0_pred)
                sim = (curr_emb * id_emb).sum()
                loss += (1 - sim) * current_id_scale

            # temporal Loss (Consistency with previous frame)
            # The code keeps temporal loss constant or ramps it similarly,
            # here it is kept constant to bind frames together throughout the process.
            if temp_scale > 0 and prev_x0_pred is not None:
                # MSE on the predicted pixels
                mse = torch.nn.functional.mse_loss(x0_pred, prev_x0_pred)
                loss += mse * temp_scale

            return loss

        edit_cond = [prompt] if model.has_conditional_generation else None

        # Run Generation
        x_edit = ddim_edit_from_noise(
            model,
            xT_base,  # We always start from the same inverted noise
            edit_cond,
            dec_pairs=dec_pairs,
            device="cuda",
            cfg_schedule=cfg_sched,
            guidance_fn=_video_guidance,
        )

        # Store prediction for next frame's guidance
        # Note: x_edit is the final result of this frame
        prev_x0_pred = x_edit.detach().clone()

        # Post-process
        edited_aligned_rgb = postprocess_from_model(
            x_edit, renormalise=model.renormalise
        )
        out_bgr = aligner.unalign_and_paste(edited_aligned_rgb, bgr, meta, feather=10)

        # Convert BGR to RGB for GIF/Video
        frames.append(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        print(f"Frame {i+1}/{num_frames} done.")

    # Save Video
    imageio.mimsave(output_path, frames, fps=8)
    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument(
        "--mode",
        type=str,
        default="image",
        choices=["image", "video"],
        help="Output mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/diffusion_celeb_smiling.yaml",
        help="Model config path",
    )
    parser.add_argument("--image_size", type=int, default=64, help="Model image size")
    parser.add_argument("--prompt", type=str, default="smiling", help="Editing prompt")
    parser.add_argument("--T_test", type=int, default=50)
    parser.add_argument("--step_percent", type=float, default=0.4)
    parser.add_argument(
        "--id_scale",
        type=float,
        default=0.0,
        help="Identity guidance scale (e.g. 50.0)",
    )
    parser.add_argument(
        "--temp_scale",
        type=float,
        default=100.0,
        help="Temporal guidance scale for video",
    )
    parser.add_argument("--edit_mode", type=str, default="invert", help="Edit mode")

    args = parser.parse_args()

    assert os.path.isfile(args.config), f"Config file {args.config} not found"
    assert os.path.isfile(args.input), f"Input image file {args.input} not found"
    filename, ext = os.path.splitext(os.path.basename(args.input))
    os.makedirs("outputs_smiling", exist_ok=True)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Setup model
    model, _ = get_model(
        cfg["model"],
        None,
        image_size=(args.image_size, args.image_size, 3),
        build_ema=False,
    )

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
    verifier = FaceVerifier(device="cuda")

    if args.mode == "image":
        out, sim_score = edit_smile(
            model,
            aligner,
            verifier,
            args.input,
            prompt=args.prompt,
            T_test=args.T_test,
            step_percent=args.step_percent,
            id_scale=args.id_scale,
            edit_mode=args.edit_mode,
        )
        print(f"Face similarity: {sim_score:.4f}")
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(cv2.imread(args.input), cv2.COLOR_BGR2RGB))
        plt.title("Input")
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title(
            f"Prompt: {args.prompt} \n Similarity: {sim_score:.4f} (using FaceNet)"
        )
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        plt.tight_layout()
        plt.savefig(
            f"outputs_smiling/comparison_{filename}.png", dpi=200, bbox_inches="tight"
        )  # saves the whole figure
        plt.show()
        cv2.imwrite(f"outputs_smiling/{filename}_edit_{args.id_scale}.jpg", out)

    elif args.mode == "video":
        make_progressive_video(
            model,
            aligner,
            verifier,
            args.input,
            prompt=args.prompt,
            output_path=f"outputs_smiling/{filename}_progression.gif",
            T_test=args.T_test,
            step_percent=args.step_percent,
            id_scale=args.id_scale,
            temp_scale=args.temp_scale,
        )
