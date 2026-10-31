"""완전히 재시작 가능한 체크포인트 — 모델, 옵티마이저, 스텝, RNG 상태, 데이터 순서
시드까지 한 파일에 담는다.

Colab 무료 티어는 예고 없이 끊기고 백그라운드 실행이 안 되므로, "사람이 다시 접속해서
셀을 또 실행하면 정확히 그 지점부터 이어지는 것"이 핵심 요구사항이다. 그래서
`train_ard.py`의 `policy.save_pretrained(ckpt_dir)`(모델 가중치만, HF 스타일 디렉터리)
보다 더 많은 정보가 필요하다 — 이 모듈은 그 전부를 **단일 파일**로 저장한다. 두
메커니즘은 서로 독립적이고 섞이지 않는다(파일 이름 규칙이 다르다 — `ckpt_step_*.pt`
vs `step_*/`).

**데이터 순서 재현에 대한 중요한 한계(정직하게 밝힘)**: `DataLoader(shuffle=True)`는
자체로는 "어느 배치까지 썼는지"를 저장할 수 있는 상태를 노출하지 않는다. 이 모듈은
`step`과 `data_seed`만 저장하고, 재시작 시 호출자(`run_colab_training.py`)가 같은
`data_seed`로 `torch.Generator`를 다시 만들어 **처음부터 다시 셔플한 뒤 `step %
steps_per_epoch`개 배치를 건너뛰어서(fast-forward)** 같은 지점으로 돌아간다 — 이건
"그 배치들을 다시 로드만 하고 버린다"는 뜻이라, 데이터셋이 크거나 step이 에포크
후반이면 재시작마다 그만큼 느려진다(완전한 無비용 재개가 아니다 — 알려진 한계,
`docs/colab_free_tier.md` 참고). 결정론적이라는 것(정확히 같은 지점으로 돌아간다는 것)
자체는 `tests/test_colab_checkpoint.py`로 보장한다.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

_CKPT_RE = re.compile(r"^ckpt_step_(\d+)\.pt$")


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    data_seed: int,
    scheduler=None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    extra: dict | None = None,
) -> Path:
    """`path`에 전체 학습 상태를 저장한다. 임시 파일에 먼저 쓰고 성공한 뒤에만
    `path`로 rename한다 — 쓰는 도중 Colab 세션이 끊겨도 `path` 자체는 이전 체크포인트
    상태 그대로 남아 있거나(아직 rename 전) 새 체크포인트로 완전히 교체돼 있거나
    둘 중 하나만 가능하다(반쪽만 써진 손상 파일이 "최신 체크포인트"로 보이는 경우를
    막는다)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "data_seed": data_seed,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": _rng_state(),
        "extra": extra or {},
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def load_checkpoint(
    path: str | Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """체크포인트를 불러와 `policy`/`optimizer`/(있으면) `scheduler`/`scaler`와 전역 RNG
    상태를 그 자리에서 복원한다. 반환값의 `step`/`data_seed`/`extra`를 호출자가 학습
    루프 재시작 지점으로 쓴다."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    policy.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    _restore_rng_state(payload["rng_state"])
    return payload


def find_latest_checkpoint(directory: str | Path) -> Path | None:
    directory = Path(directory)
    if not directory.exists():
        return None
    best_step = -1
    best_path = None
    for p in directory.glob("ckpt_step_*.pt"):
        m = _CKPT_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best_path = p
    return best_path


def checkpoint_path_for_step(directory: str | Path, step: int) -> Path:
    return Path(directory) / f"ckpt_step_{step:08d}.pt"


def prune_old_checkpoints(directory: str | Path, keep_last_n: int = 3) -> list[Path]:
    """가장 최근 `keep_last_n`개만 남기고 나머지 `ckpt_step_*.pt`를 지운다. Drive 용량이
    한정적이므로(Colab 무료 티어) 체크포인트가 쌓이는 걸 막는다. 지운 경로 목록을
    반환한다(로깅용)."""
    directory = Path(directory)
    if not directory.exists():
        return []
    entries = []
    for p in directory.glob("ckpt_step_*.pt"):
        m = _CKPT_RE.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    entries.sort(key=lambda t: t[0])
    to_delete = [p for _, p in entries[:-keep_last_n]] if keep_last_n > 0 else [p for _, p in entries]
    for p in to_delete:
        p.unlink(missing_ok=True)
    return to_delete
