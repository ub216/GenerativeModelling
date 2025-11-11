from typing import Dict

import torch
from torch import optim
from torch.amp import GradScaler


class OptimizerManager:
    def __init__(
        self,
        optimizers: Dict[str, optim.Optimizer],
        use_scaler: bool = True,
        accumulate_steps: int = 1,
    ) -> None:
        """
        Helper for managing optimizer + grad scaler updates with accumulation.

        optimizers: dict of named optimizers.
        use_scaler: whether to use AMP gradient scaling.
        accumulate_steps: gradient accumulation steps before an update.
        """
        self.optimizers = optimizers
        self.scalers = (
            {key: GradScaler() for key in optimizers.keys()}
            if use_scaler
            else {key: None for key in optimizers.keys()}
        )
        self.accumulate_steps: int = accumulate_steps
        self._step_count: int = 0

    def backward(self, losses: Dict[str, torch.Tensor]) -> None:
        """
        Backward pass with optional gradient scaling.

        losses: dict of losses with same keys as optimizers.
        """
        assert (
            losses.keys() == self.optimizers.keys()
        ), f"Loss keys {losses.keys()} must match optimizer keys {self.optimizers.keys()}"

        scaled_losses = {
            key: val.mean() / self.accumulate_steps for key, val in losses.items()
        }

        for key in self.optimizers.keys():
            if self.scalers[key] is not None:
                self.scalers[key].scale(scaled_losses[key]).backward()
            else:
                scaled_losses[key].backward()

    def step(self, force: bool = False) -> None:
        """
        Perform optimizer step if accumulation reached or forced.

        force: if True, always perform update (e.g. at epoch end).
        """
        self._step_count += 1

        if not force and self._step_count % self.accumulate_steps != 0:
            return

        for key, opt in self.optimizers.items():
            if self.scalers[key] is not None:
                self.scalers[key].step(opt)
                self.scalers[key].update()
            else:
                opt.step()

    def zero_grad(self) -> None:
        """
        Zero gradients for all optimizers.
        """
        for key, opt in self.optimizers.items():           
            opt.zero_grad(set_to_none=True)

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
