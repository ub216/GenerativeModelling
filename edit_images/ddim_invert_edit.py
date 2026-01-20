from typing import Callable, List, Optional, Tuple

import torch


def make_ddim_time_pairs(train_T: int, T_test: int, device) -> List[Tuple[int, int]]:
    # Create a schedule from 0 up to T-1
    times = torch.linspace(
        0, train_T - 1, T_test + 1, dtype=torch.long, device=device
    ).tolist()

    # Inversion: (0 -> 20), (20 -> 40) ... (960 -> 980)
    # Moving from Clean (Alpha ~1) to Noisy (Alpha ~0)
    inc = list(zip(times[:-1], times[1:]))

    # Editing/Sampling: (980 -> 960), (960 -> 940) ... (20 -> 0)
    # Moving from Noisy (Alpha ~0) to Clean (Alpha ~1)
    dec = list(zip(reversed(times[1:]), reversed(times[:-1])))

    return inc, dec


@torch.no_grad()
def predict_eps_cfg(
    model,
    x_t: torch.Tensor,
    t_batch: torch.Tensor,
    cond: Optional[List[str]],
    cfg_scale: float,
) -> torch.Tensor:
    """
    Model's UNet returns (eps, _). This wraps CFG sampler.
    """
    if model.has_conditional_generation:
        assert cond is not None and len(cond) == x_t.shape[0]
        u = [""] * x_t.shape[0]
        eps_all, _ = model.unet(
            torch.cat([x_t, x_t], dim=0),
            torch.cat([t_batch, t_batch], dim=0),
            conditioning=cond + u,
        )
        e_cond, e_uncond = eps_all.chunk(2, dim=0)
        return e_uncond + cfg_scale * (e_cond - e_uncond)
    else:
        eps, _ = model.unet(x_t, t_batch, conditioning=None)
        return eps


@torch.no_grad()
def ddim_step(
    x_t: torch.Tensor,
    eps_theta: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    alpha_bar_next: torch.Tensor,
    dynamic_threshold_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    # clamp alpha to avoid division by near-zero at the noisy end of the schedule
    eps_safe = 1e-5
    a_t_safe = alpha_bar_t.clamp(min=eps_safe)

    # predict x0 (the "clean" image estimate)
    pred_x0 = (x_t - torch.sqrt(1.0 - alpha_bar_t) * eps_theta) / torch.sqrt(a_t_safe)

    if dynamic_threshold_fn is not None:
        pred_x0 = dynamic_threshold_fn(pred_x0)
    # Also clamp x0 to prevent extreme outliers during the ODE walk
    pred_x0 = pred_x0.clamp(-1.0, 1.0)

    # compute the direction pointing to x_t
    dir_xt = torch.sqrt(1.0 - alpha_bar_next) * eps_theta

    # compute x_next
    x_next = torch.sqrt(alpha_bar_next) * pred_x0 + dir_xt
    return x_next


@torch.no_grad()
def ddim_invert(
    model,
    x0: torch.Tensor,  # (1,C,H,W) in model space (maybe [-1,1] if renormalise)
    conditioning: Optional[List[str]],  # usually [""] for inversion
    inc_pairs: List[Tuple[int, int]],
    device: str,
) -> torch.Tensor:
    """
    Produces x_T (noisy latent) via deterministic DDIM inversion along the same skip schedule.
    """
    model.unet.eval()

    x_t = x0.to(device)
    for t_curr, t_next in inc_pairs:
        t_batch = torch.full((x_t.shape[0],), t_curr, device=device, dtype=torch.long)
        eps = predict_eps_cfg(model, x_t, t_batch, conditioning, cfg_scale=0.0)

        a_t = model.train_alphas_cumprod[t_curr]
        a_next = model.train_alphas_cumprod[t_next]

        x_t = ddim_step(
            x_t,
            eps,
            a_t,
            a_next,
            dynamic_threshold_fn=None,
        )
    return x_t


@torch.no_grad()
def ddim_edit_from_noise(
    model,
    x_T: torch.Tensor,  # (1,C,H,W)
    conditioning: Optional[List[str]],  # e.g. ["smiling"]
    dec_pairs: List[Tuple[int, int]],
    device: str,
    cfg_schedule: Callable[[int, int], float],  # (step_idx, num_steps) -> scale
) -> torch.Tensor:
    """
    Deterministic DDIM reverse starting from x_T using skip schedule.
    """
    model.unet.eval()

    x_t = x_T.to(device)
    num_steps = len(dec_pairs)

    for i, (t_curr, t_next) in enumerate(dec_pairs):
        t_batch = torch.full((x_t.shape[0],), t_curr, device=device, dtype=torch.long)
        cfg = float(cfg_schedule(i, num_steps))
        eps = predict_eps_cfg(model, x_t, t_batch, conditioning, cfg_scale=cfg)

        a_t = model.train_alphas_cumprod[t_curr]
        a_next = (
            model.train_alphas_cumprod[t_next]
            if t_next >= 0
            else torch.tensor(1.0, device=device)
        )

        x_t = ddim_step(
            x_t,
            eps,
            a_t,
            a_next,
            dynamic_threshold_fn=None,  # model._dynamic_threshold,
        )
    return x_t


def linear_cfg_ramp(cfg_start: float, cfg_end: float):
    def _sched(i: int, n: int) -> float:
        if n <= 1:
            return cfg_end
        w = i / (n - 1)
        return (1 - w) * cfg_start + w * cfg_end

    return _sched
