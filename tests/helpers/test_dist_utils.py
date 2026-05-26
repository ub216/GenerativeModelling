import os
import socket

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from helpers.distributed_utils import (
    cleanup_distributed_training,
    get_rank,
    get_world_size,
    init_distributed_training,
    is_distributed,
    is_main_process,
    reduce_tensor,
    unwrap_model,
)

_WORLD_SIZE = 2


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


def _worker_rank_world_size(rank: int, world_size: int, port: int) -> None:
    _setup(rank, world_size, port)
    assert get_rank() == rank
    assert get_world_size() == world_size
    assert is_distributed()
    assert is_main_process() == (rank == 0)
    _teardown()


def _worker_reduce_mean(rank: int, world_size: int, port: int) -> None:
    _setup(rank, world_size, port)
    # rank 0 contributes 0.0, rank 1 contributes 1.0 → mean = 0.5
    t = torch.tensor(float(rank))
    result = reduce_tensor(t, op="mean")
    assert abs(result.item() - 0.5) < 1e-6, f"Expected 0.5, got {result.item()}"
    _teardown()


def _worker_unwrap_model(rank: int, world_size: int, port: int) -> None:
    _setup(rank, world_size, port)

    raw = nn.Linear(4, 2)
    ddp = DDP(raw)

    assert unwrap_model(ddp) is raw, f"Rank {rank}: unwrap_model should return the inner module"
    assert unwrap_model(raw) is raw, f"Rank {rank}: unwrap_model on a plain model should be a no-op"

    # Parameter names on the unwrapped model must not carry the 'module.' prefix
    names = [n for n, _ in unwrap_model(ddp).named_parameters()]
    assert all(
        "module." not in n for n in names
    ), f"Rank {rank}: unwrapped model has unexpected 'module.' prefix in param names: {names}"

    _teardown()


def test_rank_and_world_size_detection():
    port = _find_free_port()
    mp.spawn(_worker_rank_world_size, args=(_WORLD_SIZE, port), nprocs=_WORLD_SIZE, join=True)


def test_reduce_tensor_averages_across_ranks():
    port = _find_free_port()
    mp.spawn(_worker_reduce_mean, args=(_WORLD_SIZE, port), nprocs=_WORLD_SIZE, join=True)


def test_unwrap_model_strips_ddp():
    port = _find_free_port()
    mp.spawn(_worker_unwrap_model, args=(_WORLD_SIZE, port), nprocs=_WORLD_SIZE, join=True)
