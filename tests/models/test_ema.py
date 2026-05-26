import os
import socket

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from helpers.distributed_utils import cleanup_distributed_training, init_distributed_training
from models.ema import EMAModel


def _linear(val=1.0):
    """Return a freshly initialised Linear(4,2) with all weights set to val."""
    m = nn.Linear(4, 2)
    nn.init.constant_(m.weight, val)
    nn.init.zeros_(m.bias)
    return m


class TestEMAModelInit:
    def test_shadow_params_have_no_grad(self):
        ema = EMAModel(_linear(), decay=0.9)
        for p in ema.model.parameters():
            assert not p.requires_grad

    def test_model_in_eval_mode(self):
        ema = EMAModel(_linear(), decay=0.9)
        assert not ema.model.training


class TestEMAUpdate:
    def test_decay_one_shadow_unchanged(self):
        """decay=1 → shadow never moves regardless of online values."""
        shadow_nn = _linear(val=1.0)
        ema = EMAModel(shadow_nn, decay=1.0)
        shadow_before = ema.model.weight.data.clone()

        online = _linear(val=999.0)
        ema.update(online)

        assert torch.equal(ema.model.weight.data, shadow_before)

    def test_decay_zero_shadow_copies_online(self):
        """decay=0 → shadow becomes identical to online after one update."""
        ema = EMAModel(_linear(val=0.0), decay=0.0)
        online = _linear(val=7.0)
        ema.update(online)
        assert torch.allclose(ema.model.weight.data, online.weight.data)

    def test_update_interpolates_correctly(self):
        """shadow = decay * shadow + (1 - decay) * online."""
        decay = 0.9
        ema = EMAModel(_linear(val=1.0), decay=decay)
        online = _linear(val=5.0)

        shadow_before = ema.model.weight.data.clone()
        ema.update(online)

        expected = decay * shadow_before + (1 - decay) * online.weight.data
        assert torch.allclose(ema.model.weight.data, expected, atol=1e-6)

    def test_repeated_updates_converge_to_online(self):
        """After many updates (decay=0.5), shadow converges toward online."""
        ema = EMAModel(_linear(val=0.0), decay=0.5)
        # zero-init shadow's weight manually (EMAModel shares the reference)
        nn.init.zeros_(ema.model.weight)

        online = _linear(val=10.0)
        for _ in range(30):
            ema.update(online)

        assert torch.allclose(ema.model.weight.data, online.weight.data, atol=0.02)

    def test_update_does_not_affect_online_model(self):
        """update() must not modify the online model's parameters."""
        online = _linear(val=5.0)
        ema = EMAModel(_linear(val=1.0), decay=0.9)
        weight_before = online.weight.data.clone()
        ema.update(online)
        assert torch.equal(online.weight.data, weight_before)

    def test_update_handles_multiple_parameter_groups(self):
        """All named params (weight + bias) must be updated."""
        ema = EMAModel(_linear(val=1.0), decay=0.0)
        online = _linear(val=5.0)
        nn.init.constant_(online.bias, 3.0)
        ema.update(online)
        assert torch.allclose(ema.model.weight.data, online.weight.data)
        assert torch.allclose(ema.model.bias.data, online.bias.data)


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


def _worker_ema_ddp_update(rank: int, world_size: int, port: int) -> None:
    _setup(rank, world_size, port)

    online_nn = _linear(val=5.0)
    ema = EMAModel(_linear(val=1.0), decay=0.0)

    ddp_model = DDP(online_nn)
    # update() must unwrap DDP and resolve parameter names without "module." prefix
    ema.update(ddp_model)

    # decay=0 → shadow must exactly equal the online weights after one update
    assert torch.allclose(
        ema.model.weight.data, online_nn.weight.data
    ), f"Rank {rank}: shadow weight mismatch after DDP-wrapped EMA update"
    assert torch.allclose(
        ema.model.bias.data, online_nn.bias.data
    ), f"Rank {rank}: shadow bias mismatch after DDP-wrapped EMA update"

    _teardown()


class TestEMAWithDDP:
    def test_update_unwraps_ddp_and_resolves_param_names(self):
        port = _find_free_port()
        mp.spawn(_worker_ema_ddp_update, args=(2, port), nprocs=2, join=True)
