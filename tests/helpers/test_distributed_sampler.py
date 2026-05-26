import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from helpers.distributed_utils import cleanup_distributed_training, init_distributed_training

_WORLD_SIZE = 2
_DATASET_SIZE = 100


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


def _worker_disjoint_indices(rank: int, world_size: int, port: int) -> None:
    _setup(rank, world_size, port)

    dataset = TensorDataset(torch.arange(_DATASET_SIZE))
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=10, sampler=sampler)

    sampler.set_epoch(0)
    local_indices = torch.cat([x for (x,) in loader])

    # Gather every rank's indices onto every rank, then verify no duplicates
    gathered = [torch.zeros_like(local_indices) for _ in range(world_size)]
    dist.all_gather(gathered, local_indices)
    all_indices = torch.cat(gathered)
    assert all_indices.unique().numel() == all_indices.numel(), f"Rank {rank}: indices are not disjoint across ranks"

    _teardown()


def _worker_set_epoch_changes_order(rank: int, world_size: int, port: int) -> None:
    _setup(rank, world_size, port)

    dataset = TensorDataset(torch.arange(_DATASET_SIZE))
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)

    sampler.set_epoch(0)
    epoch0 = list(sampler)

    sampler.set_epoch(1)
    epoch1 = list(sampler)

    assert epoch0 != epoch1, f"Rank {rank}: set_epoch did not change index ordering"

    _teardown()


def _worker_drop_last_equal_length(rank: int, world_size: int, port: int) -> None:
    """All ranks must receive the same number of samples when drop_last=True."""
    _setup(rank, world_size, port)

    # Odd-sized dataset so without drop_last ranks would get different counts
    dataset = TensorDataset(torch.arange(101))
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=5, sampler=sampler)

    local_count = torch.tensor(sum(x.shape[0] for (x,) in loader))
    gathered = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(gathered, local_count)

    counts = [t.item() for t in gathered]
    assert len(set(counts)) == 1, f"Unequal sample counts across ranks: {counts}"

    _teardown()


def test_indices_are_disjoint_across_ranks():
    port = _find_free_port()
    mp.spawn(_worker_disjoint_indices, args=(_WORLD_SIZE, port), nprocs=_WORLD_SIZE, join=True)


def test_set_epoch_changes_index_ordering():
    port = _find_free_port()
    mp.spawn(_worker_set_epoch_changes_order, args=(_WORLD_SIZE, port), nprocs=_WORLD_SIZE, join=True)


def test_drop_last_gives_equal_counts_across_ranks():
    port = _find_free_port()
    mp.spawn(_worker_drop_last_equal_length, args=(_WORLD_SIZE, port), nprocs=_WORLD_SIZE, join=True)
