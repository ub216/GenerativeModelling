import math
from typing import List, Optional, Tuple

import torch
from loguru import logger
from torch.func import jvp

import helpers.custom_types as custom_types
from helpers.diffusion_utils import drop_condition
from models.backbone.unet import UNet
from models.flow import FlowModel


# -----------------------------
# MeanFlow model implementation: https://arxiv.org/pdf/2505.13447
# -----------------------------
class MeanFlowModel(FlowModel):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = (1, 2, 4),
        num_blocks: int | List[int] = [1, 2, 2],
        time_emb_dim: int = 128,
        timesteps: int = 1000,
        device: custom_types.DeviceType = "cuda",
        test_timesteps: Optional[int] = None,
        text_emb_dim: Optional[int] = None,
        drop_condition_ratio: float = 0.1,  # paper Sec 4.2: 10% unconditional training
        sample_condition_weight: float = 1.0,  # ω in Eq 21; effective scale ω'=ω/(1-κ)=2.0 (ImageNet B/2 Table 4)
        renormalize: bool = False,
        use_attention: bool = False,
        same_time_ratio: float = 0.75,  # fraction of batch where r=t is forced; ImageNet B/2: 75% (Table 4)
        kappa: float = 0.5,  # κ in Eq 21; CFG mixing weight (ImageNet B/2 Table 4)
        logit_sigma: float = 1.0,  # logit-normal σ; ImageNet B/2: lognorm(-0.4, 1.0) (Table 4)
        logit_mu: float = -0.4,  # logit-normal μ; concentrates mass away from t=0 boundary
        use_finite_diff: bool = False,  # if True, estimate du/dt via finite differences (~half peak memory vs JVP)
        *args,
        **kwargs,
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_mults=channel_mults,
            num_blocks=num_blocks,
            time_emb_dim=time_emb_dim,
            text_emb_dim=text_emb_dim,
            timesteps=timesteps,
            device=device,
            test_timesteps=test_timesteps,
            drop_condition_ratio=drop_condition_ratio,
            sample_condition_weight=sample_condition_weight,
            renormalize=renormalize,
            use_attention=use_attention,
            *args,
            **kwargs,
        )
        # Replace the single-time UNet created by FlowModel with a dual-time one for MeanFlow
        self.unet = UNet(
            in_channels,
            base_channels,
            channel_mults,
            num_blocks=num_blocks,
            time_emb_dim=(time_emb_dim, time_emb_dim),
            text_emb_dim=text_emb_dim,
            device=device,
            use_attention=use_attention,
        )

        self.same_time_ratio = same_time_ratio
        self.kappa = kappa
        self.logit_sigma = logit_sigma
        self.logit_mu = logit_mu
        self.use_finite_diff = use_finite_diff
        self.has_conditional_generation = True if text_emb_dim is not None else False
        if self.has_conditional_generation:
            logger.info("Created a conditioned mean flow matching model")
        else:
            logger.info("Created an unconditioned mean flow matching model")

    def forward(
        self,
        x0: torch.Tensor,
        time_steps: Optional[torch.Tensor] = None,
        x1: Optional[torch.Tensor] = None,
        conditioning: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.valid_input_combination(conditioning):
            raise ValueError("Invalid input combination")

        if self.renormalize:
            x0 = x0 * 2.0 - 1.0  # to [-1, 1]

        b = x0.shape[0]
        if time_steps is None:
            # Logit-normal sampling (Sec 3.3): sigmoid maps a normal onto (0,1) with more mass near
            # 0 and 1 than uniform sampling, which focuses training on the harder boundary regions.
            time_steps = torch.sigmoid(torch.randn(b, 2, device=x0.device) * self.logit_sigma + self.logit_mu)
            time_steps = torch.sort(time_steps, dim=1, descending=True).values  # ensure t >= r: col0=t, col1=r
            same_time = torch.rand(b, device=x0.device) < self.same_time_ratio
            # Boundary condition (Eq 7): u(z_t, t, t) = v(z_t, t). Setting r=t forces the network
            # to match the instantaneous velocity at zero interval, stabilising early training.
            # torch.where instead of in-place masked scatter: index_put_ with a data-dependent
            # boolean mask is not capturable by CUDA graphs (reduce-overhead compile mode).
            time_steps = torch.where(same_time.unsqueeze(1), time_steps[:, 0:1].expand_as(time_steps), time_steps)
        else:
            # When explicit time_steps are provided (e.g. tests, evaluation), derive the boundary
            # mask from whether t == r rather than sampling it randomly.
            same_time = torch.isclose(time_steps[:, 0], time_steps[:, 1])

        if x1 is None:
            x1 = torch.randn_like(x0)
        if conditioning is not None:
            conditioning = drop_condition(conditioning, self.drop_condition_ratio)

        intermidiate = self.q_sample(x0, time_steps[:, 0], x1)  # zt = (1-t) * z0 + t * noise
        flow = x1 - x0
        text_emb = (
            self.unet.text_model(conditioning)
            if conditioning is not None and self.unet.text_model is not None
            else None
        )
        if self.has_conditional_generation:
            with torch.no_grad():
                same_timesteps = torch.stack([time_steps[:, 0], torch.zeros_like(time_steps[:, 0])], dim=1)
                # Batch cond and uncond into one forward pass. Empty-string embedding for uncond
                # matches training behaviour (drop_condition replaces strings with "", not None).
                empty_emb = self.unet.text_model([""] * b)
                batched_z = torch.cat([intermidiate, intermidiate], dim=0)
                batched_t = torch.cat([same_timesteps, same_timesteps], dim=0)
                batched_emb = torch.cat([text_emb, empty_emb], dim=0)
                u_both, _ = self.unet(batched_z, batched_t, text_emb=batched_emb)
                u_cond, u_uncond = u_both.chunk(2)
                flow = (
                    self.sample_condition_weight * flow
                    + self.kappa * u_cond
                    + (1 - self.sample_condition_weight - self.kappa) * u_uncond
                )  # vt^cfg = w*vt + kappa*u_cond + (1-w-kappa)*u_uncond
        time_conditioning = time_steps.clone()
        # Network is conditioned on (t, Δt=t-r) rather than (t, r) so the Δt=0 boundary is a
        # single clean value regardless of t, making it easier for the model to learn Eq 7.
        time_conditioning[:, 1] = time_steps[:, 0] - time_steps[:, 1]  # col1: r -> Δt = t - r

        def fn(z, t_cond):
            return self.unet(z, t_cond, text_emb=text_emb)[0]

        # Tangent is [1, 1], not [1, 0]. We want du/dt holding r fixed, i.e. the total derivative
        # along the trajectory. In (t, Δt) coordinates, d(Δt)/dt = d(t-r)/dt = 1 when r is fixed,
        # so both components advance together. Using [1, 0] would compute the partial w.r.t. t
        # holding Δt fixed, which is a different (and wrong) quantity.
        time_tangent = torch.ones_like(time_conditioning)
        if self.use_finite_diff:
            pred_u, dudt = self._fd_step(fn, intermidiate, time_conditioning, flow, time_tangent)
        else:
            pred_u, dudt = self._jvp_step(fn, intermidiate, time_conditioning, flow, time_tangent)

        # MeanFlow identity (Eq 6): u(z_t, r, t) = v(z_t, t) - (t-r) * du/dt
        # pred_u approximates the left-hand side; target_u is the right-hand side computed from
        # the instantaneous velocity v (=flow) and the JVP estimate of du/dt.
        target_u = flow - (time_steps[:, 0] - time_steps[:, 1]).view(-1, 1, 1, 1) * dudt.detach()
        return pred_u, target_u, same_time

    @torch.compiler.disable
    def _jvp_step(self, fn, z, t_cond, flow, time_tangent):
        # Exact du/dt via forward-mode AD. Each intermediate activation exists as a dual
        # tensor (primal + tangent) so peak memory is ~2× a single forward pass.
        # torch._dynamo cannot trace through GradTrackingTensor — excluded from compilation.
        return jvp(fn, (z, t_cond), (flow, time_tangent))

    @torch.compiler.disable
    def _fd_step(self, fn, z, t_cond, flow, time_tangent):
        # Finite-difference approximation of du/dt along dz/dt = flow.
        # The perturbed pass runs under no_grad so activations are freed layer-by-layer,
        # giving ~half the peak memory of _jvp_step at the cost of O(eps) approximation error.
        # Autocast is disabled for the perturbed pass to prevent bf16 catastrophic cancellation
        # when subtracting near-equal values; pred_u stays in the caller's AMP dtype.
        pred_u = fn(z, t_cond)
        _eps = 1e-3
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            u_eps = fn((z + _eps * flow).float(), (t_cond + _eps * time_tangent).float())
        dudt = (u_eps.to(pred_u.dtype) - pred_u.detach()) / _eps
        return pred_u, dudt

    @torch.no_grad()
    def sample_flow(
        self,
        num_samples: int,
        device: custom_types.DeviceType,
        image_size: Tuple[int, int],
        batch_size: int = 16,
        conditioning: Optional[List[str]] = None,
        dynamic_threshold: bool = True,
        threshold_coeff: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate samples by iteratively predicting the flow.
        num_samples: int
        device: torch device
        image_size: (H, W) tuple
        batch_size: int
        conditioning: input conditioning
        returns: (num_samples, C, H, W) raw tensor (not clamped or renormalized)
        """
        if not self.valid_input_combination(conditioning):
            raise ValueError("Invalid input combination")

        self.unet.eval()
        samples = []
        for idx in range(math.ceil(num_samples / batch_size)):
            cur_bs = min(batch_size, num_samples - idx * batch_size)
            x_t = torch.randn(cur_bs, self.in_channels, image_size[0], image_size[1], device=device)

            cond_batch = conditioning[idx * batch_size : idx * batch_size + cur_bs] if conditioning else None
            # Encode text once per sample batch, not once per diffusion step
            text_emb_cache = (
                self.unet.text_model(cond_batch)
                if cond_batch is not None and self.unet.text_model is not None
                else None
            )

            # same_time_ratio==1 means the model was trained only on boundary conditions
            # (r=t, Δt=0), so it only knows instantaneous velocity — query with Δt=0.
            # Otherwise use test_delta so the network predicts mean velocity over each step.
            sample_delta = 0.0 if self.same_time_ratio == 1.0 else self.test_delta

            for t in range(self.test_timesteps, 0, -1):  # t is T, T-1, ..., 1
                t_val = t / self.test_timesteps
                vec_t = torch.stack(
                    [
                        torch.full((cur_bs,), t_val, device=device),
                        torch.full((cur_bs,), sample_delta, device=device),
                    ],
                    dim=1,
                )  # (B, 2): current time and Δt

                u_pred, _ = self.unet(x_t, vec_t, text_emb=text_emb_cache)

                # pred_x0 = z_t - t*v(z_t,t): the predicted clean image, not the next ODE state.
                # Dynamic thresholding must operate on the full x0 estimate; using test_delta
                # (step size) instead of t_val would threshold the tiny next-step increment,
                # compressing noise-scale states and corrupting all subsequent steps.
                pred_x0 = x_t - t_val * u_pred
                if dynamic_threshold:
                    pred_x0 = self._dynamic_threshold(pred_x0, c=threshold_coeff)
                else:
                    pred_x0 = pred_x0.clamp(-threshold_coeff, threshold_coeff)

                # Back out corrected velocity from thresholded x0 and take a small Euler step
                x_t = x_t - self.test_delta * ((x_t - pred_x0) / t_val)

            samples.append(x_t)

        self.unet.train()
        return torch.cat(samples, dim=0)[:num_samples]
