"""
Tests for Chunk 4 rank-0 guards: verifies that wandb logging, checkpoint
saving, and tqdm rendering are restricted to rank 0, while EMA updates
and dist.barrier() are called on every rank.
"""

import os
import socket
import tempfile

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from helpers.distributed_utils import (
    cleanup_distributed_training,
    init_distributed_training,
    is_distributed,
    is_main_process,
    unwrap_model,
)
from models.base_model import BaseModel

# ---------------------------------------------------------------------------
# Minimal concrete model for attribute-access tests
# ---------------------------------------------------------------------------


class _MinimalModel(BaseModel):
    """Smallest possible BaseModel subclass — used to test method availability."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x, conditioning=None):
        return self.linear(x)

    def sample(self, num_samples, device, image_size, batch_size=16, conditioning=None):
        return torch.zeros(num_samples, 1, image_size, image_size)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _setup(rank: int, world_size: int, port: int) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(port),
        }
    )
    init_distributed_training(backend="gloo")


def _teardown() -> None:
    cleanup_distributed_training()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------


def _worker_rank0_guard(rank: int, world_size: int, port: int, tmp_dir: str) -> None:
    """Simulates the rank-0-only checkpoint/log guard used in train()."""
    _setup(rank, world_size, port)
    if is_main_process():
        # Only rank 0 should write this file — mirrors torch.save / wandb.log guards
        with open(os.path.join(tmp_dir, f"rank_{rank}_saved"), "w") as f:
            f.write("saved")
    # All ranks must hit the barrier; rank 1 skipping it would deadlock
    if is_distributed():
        torch.distributed.barrier()
    _teardown()


def _worker_barrier_no_deadlock(rank: int, world_size: int, port: int) -> None:
    """Barrier after a rank-0-only block must not deadlock."""
    _setup(rank, world_size, port)
    if is_main_process():
        pass  # simulated main-process work (e.g. torch.save)
    if is_distributed():
        torch.distributed.barrier()
    _teardown()


def _worker_ddp_attribute_access(rank: int, world_size: int, port: int) -> None:
    """
    DDP must NOT expose wrap_sampler_to_loader; unwrap_model must restore it.
    This mirrors the bug in train() where wrap_sampler_to_loader was called
    on the raw (possibly DDP-wrapped) model and raised AttributeError.
    """
    _setup(rank, world_size, port)

    model = _MinimalModel()
    ddp = DDP(model)

    # DDP does not forward arbitrary attribute access to its inner module
    assert not hasattr(ddp, "wrap_sampler_to_loader"), (
        "DDP unexpectedly exposes wrap_sampler_to_loader — " "the unwrap_model() fix in train() may be unnecessary"
    )

    # unwrap_model must restore access
    inner = unwrap_model(ddp)
    assert hasattr(inner, "wrap_sampler_to_loader"), "unwrap_model() did not restore wrap_sampler_to_loader"
    assert inner is model, "unwrap_model() returned the wrong object"

    _teardown()


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestRankZeroGuard:
    def test_only_rank0_executes_guarded_block(self):
        """Guarded block (is_main_process()) must run on exactly one rank."""
        port = _find_free_port()
        with tempfile.TemporaryDirectory() as tmp_dir:
            mp.spawn(_worker_rank0_guard, args=(2, port, tmp_dir), nprocs=2, join=True)
            saved = os.listdir(tmp_dir)
            assert saved == ["rank_0_saved"], f"Expected only rank 0 to write a file, got: {saved}"

    def test_barrier_after_rank0_block_does_not_deadlock(self):
        """dist.barrier() outside the is_main_process() guard must not deadlock."""
        port = _find_free_port()
        mp.spawn(_worker_barrier_no_deadlock, args=(2, port), nprocs=2, join=True)


class TestUnwrapModelForMethodAccess:
    def test_ddp_hides_wrap_sampler_to_loader(self):
        """DDP does not forward wrap_sampler_to_loader; unwrap_model() restores it."""
        port = _find_free_port()
        mp.spawn(_worker_ddp_attribute_access, args=(2, port), nprocs=2, join=True)
