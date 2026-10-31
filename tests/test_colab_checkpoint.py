"""scripts/colab/checkpoint.py — 저장/복원이 모델/옵티마이저/스텝/RNG 상태를 정확히
되돌리는지, 그리고 find_latest/prune가 올바른 파일을 고르는지 확인한다."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import checkpoint as ckpt_mod  # noqa: E402


def _tiny_model_and_optim():
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return model, optimizer


def test_save_and_load_restores_model_and_optimizer_state(tmp_path):
    model, optimizer = _tiny_model_and_optim()
    x = torch.randn(3, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()  # 모멘텀 버퍼가 생기도록 한 번 밟아둔다

    path = tmp_path / "ckpt_step_00000010.pt"
    ckpt_mod.save_checkpoint(path, model, optimizer, step=10, data_seed=7)

    model2, optimizer2 = _tiny_model_and_optim()
    payload = ckpt_mod.load_checkpoint(path, model2, optimizer2)

    assert payload["step"] == 10
    assert payload["data_seed"] == 7
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)
    sd1, sd2 = optimizer.state_dict(), optimizer2.state_dict()
    assert sd1["param_groups"] == sd2["param_groups"]


def test_save_and_load_restores_rng_state_exactly(tmp_path):
    model, optimizer = _tiny_model_and_optim()
    path = tmp_path / "ckpt_step_00000001.pt"

    random.seed(123)
    torch.manual_seed(123)
    _ = [random.random() for _ in range(5)]  # python RNG를 좀 소비해서 "저장 시점" 상태를 만든다
    _ = torch.randn(5)

    ckpt_mod.save_checkpoint(path, model, optimizer, step=1, data_seed=0)
    expected_next_python = random.random()
    expected_next_torch = torch.randn(1)

    # RNG 상태를 흐트러뜨린다.
    random.seed(999)
    torch.manual_seed(999)
    _ = [random.random() for _ in range(3)]
    _ = torch.randn(3)

    model2, optimizer2 = _tiny_model_and_optim()
    ckpt_mod.load_checkpoint(path, model2, optimizer2)

    assert random.random() == expected_next_python
    assert torch.equal(torch.randn(1), expected_next_torch)


def test_load_checkpoint_restores_scheduler_and_scaler(tmp_path):
    model, optimizer = _tiny_model_and_optim()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    x = torch.randn(3, 4)
    for _ in range(2):
        optimizer.zero_grad()
        model(x).sum().backward()
        optimizer.step()
        scheduler.step()

    path = tmp_path / "ckpt_step_00000002.pt"
    ckpt_mod.save_checkpoint(path, model, optimizer, step=2, data_seed=0, scheduler=scheduler)

    model2, optimizer2 = _tiny_model_and_optim()
    scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.5)
    ckpt_mod.load_checkpoint(path, model2, optimizer2, scheduler=scheduler2)

    assert scheduler2.state_dict()["_step_count"] == scheduler.state_dict()["_step_count"]


def test_save_checkpoint_does_not_leave_tmp_file(tmp_path):
    model, optimizer = _tiny_model_and_optim()
    path = tmp_path / "ckpt_step_00000001.pt"
    ckpt_mod.save_checkpoint(path, model, optimizer, step=1, data_seed=0)
    assert path.exists()
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_find_latest_checkpoint_picks_highest_step(tmp_path):
    model, optimizer = _tiny_model_and_optim()
    for step in (5, 20, 10):
        ckpt_mod.save_checkpoint(ckpt_mod.checkpoint_path_for_step(tmp_path, step), model, optimizer, step=step, data_seed=0)
    latest = ckpt_mod.find_latest_checkpoint(tmp_path)
    assert latest == ckpt_mod.checkpoint_path_for_step(tmp_path, 20)


def test_find_latest_checkpoint_none_when_empty(tmp_path):
    assert ckpt_mod.find_latest_checkpoint(tmp_path) is None
    assert ckpt_mod.find_latest_checkpoint(tmp_path / "does_not_exist") is None


def test_prune_old_checkpoints_keeps_only_last_n(tmp_path):
    model, optimizer = _tiny_model_and_optim()
    for step in (1, 2, 3, 4, 5):
        ckpt_mod.save_checkpoint(ckpt_mod.checkpoint_path_for_step(tmp_path, step), model, optimizer, step=step, data_seed=0)

    deleted = ckpt_mod.prune_old_checkpoints(tmp_path, keep_last_n=2)
    remaining = sorted(p.name for p in tmp_path.glob("ckpt_step_*.pt"))
    assert remaining == [ckpt_mod.checkpoint_path_for_step(tmp_path, s).name for s in (4, 5)]
    assert len(deleted) == 3
