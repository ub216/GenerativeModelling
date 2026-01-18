from typing import Dict

import torch
from loguru import logger
from torch import optim
from torch.amp import GradScaler


class OptimizerManager:
    def __init__(
        self,
        optimizers: Dict[str, optim.Optimizer],
        model: torch.nn.Module,
        use_scaler: bool = True,
        accumulate_steps: int = 1,
        max_grad_norm: float = float("inf"),
    ) -> None:
        self.model = model
        self.optimizers = optimizers
        self.max_grad_norm = max_grad_norm
        self.accumulate_steps = accumulate_steps
        self._step_count = 0

        self.scalers = {
            key: GradScaler() if use_scaler else None for key in optimizers.keys()
        }
        logger.info(
            f"Initialized OptimizerManager with optimizers: {list(optimizers.keys())} and use_scaler={use_scaler}, accumulate_steps={accumulate_steps}, max_grad_norm={max_grad_norm}"
        )

    def _get_params_for_key(self, key: str):
        """
        Helper to find the right parameter group based on the optimizer key.
        """
        if key == "all":
            return self.model.parameters()
        # If key is 'gen' or 'disc', it looks for model.gen or model.disc
        module = getattr(self.model, key, None)
        if module is not None and isinstance(module, torch.nn.Module):
            return module.parameters()
        # Fallback to the optimizer's own param groups if module lookup fails
        params = []
        for group in self.optimizers[key].param_groups:
            params.extend(group["params"])
        return params

    @torch.no_grad()
    def _calculate_norm(self, key: str) -> float:
        """
        Calculates the true L2 norm for a specific optimizer's parameters.
        """
        total_norm = 0.0
        params = self._get_params_for_key(key)
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.detach().norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def step(self, force: bool = False) -> Dict[str, float]:
        """
        Perform optimizer step if accumulation reached or forced.

        force: if True, always perform update (e.g. at epoch end).
        Returns:
            Dictionary of gradient norms for logging.
        """
        self._step_count += 1
        metrics = {}

        if not force and self._step_count % self.accumulate_steps != 0:
            return metrics

        for key, opt in self.optimizers.items():
            scaler = self.scalers[key]
            params = list(self._get_params_for_key(key))

            # log gradient scale if using AMP
            if scaler is not None:
                # log current scale before update/step
                metrics[f"grads/{key}_scale"] = scaler.get_scale()
                scaler.unscale_(opt)

            # clip and get the RAW norm
            raw_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            raw_norm_value = raw_norm.item()
            metrics[f"grads/{key}_norm_raw"] = raw_norm_value

            # SAFETY CHECK: Skip the step if the gradient is Inf or NaN
            if not torch.isfinite(raw_norm):
                logger.warning(
                    f"Step {self._step_count}: Non-finite gradient detected ({raw_norm_value}). Skipping update."
                )
                if scaler is not None:
                    # tells scaler to reduce scale factor for next batch
                    scaler.update()
                opt.zero_grad(set_to_none=True)
                # ensure we still log the scale even on skip
                continue

            # log the clipped norm (After clipping)
            clipped_norm = self._calculate_norm(key)
            metrics[f"grads/{key}_norm_clipped"] = clipped_norm

            # execute weight update
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

        return metrics

    def _get_grad_norm(self, model: torch.nn.Module) -> float:
        """Helper to compute total L2 norm of gradients."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm**0.5

    def zero_grad(self) -> None:
        """
        Zero gradients for all optimizers.
        """
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)

    def backward(self, losses: Dict[str, torch.Tensor]) -> None:
        """
        Backward pass with optional gradient scaling.

        losses: dict of losses with same keys as optimizers.
        """
        for key, val in losses.items():
            loss = val.mean() / self.accumulate_steps
            if self.scalers[key] is not None:
                self.scalers[key].scale(loss).backward()
            else:
                loss.backward()

    def load_state_dict(self, state_dict: Dict[str, any]) -> None:
        """
        Load optimizer states from state_dict.
        """
        for key, opt in self.optimizers.items():
            opt.load_state_dict(state_dict[key])

    def state_dict(self) -> Dict[str, any]:
        """
        Get state dict of all optimizers.
        """
        return {key: opt.state_dict() for key, opt in self.optimizers.items()}
