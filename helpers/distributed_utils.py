import os
from typing import Tuple

import torch


def init_distributed_training(backend: str | None = None) -> Tuple[int, int]:
    """Initialize distributed training environment.

    Args:
        backend: Backend to use for distributed training (e.g., "nccl" for GPUs, "gloo" for CPUs).

    Returns:
        rank: The rank of the current process.
        world_size: The total number of processes in the distributed group.
    """
    # torchrun injects RANK; if absent we're in single-process mode — skip init
    if "RANK" not in os.environ:
        return 0, 1
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    if not is_distributed():
        torch.distributed.init_process_group(backend=backend)
    return get_rank_and_world_size()


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return torch.distributed.is_initialized()


def cleanup_distributed_training():
    """Clean up the distributed training environment."""
    if is_distributed():
        torch.distributed.destroy_process_group()


def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)."""
    if not is_distributed():
        return True
    return torch.distributed.get_rank() == 0


def get_rank() -> int:
    """Get the rank of the current process."""
    if not is_distributed():
        return 0
    return torch.distributed.get_rank()


def get_world_size() -> int:
    """Get the total number of processes in the distributed group."""
    if not is_distributed():
        return 1
    return torch.distributed.get_world_size()


def get_rank_and_world_size() -> Tuple[int, int]:
    """Get the rank and world size of the current process."""
    return get_rank(), get_world_size()


def reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """Reduce a tensor across all processes.

    Args:
        tensor: The tensor to reduce.
        op: The reduction operation to apply ("mean" or "sum").

    Returns:
        The reduced tensor (modified in-place).
    """
    if op not in ("mean", "sum"):
        raise ValueError(f"Unsupported reduction operation: {op!r}")
    if get_world_size() < 2:
        return tensor
    with torch.no_grad():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        if op == "mean":
            tensor /= get_world_size()
    return tensor
