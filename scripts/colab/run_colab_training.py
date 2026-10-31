#!/usr/bin/env python3
"""무료 Colab에서 RoCo 데이터로 SmolVLA(+ARD) 스모크/베이스라인 학습을 돌리는 진입점.

`scripts/train_ard.py`의 설정-구성/데이터셋-피처-분리 로직(`split_policy_features`,
`warn_if_action_layout_looks_wrong`)을 그대로 재사용하되, 그 스크립트의 "argparse로
SmolVLAConfig를 직접 조립"하는 부분은 **재사용하지 않는다** — `train_ard.py`는 OpenArm
8DoF 기본값(`ard_arm_dim=8` 등)을 가정하고, 이 스크립트는 `configs/presets/roco_board.yaml`
프리셋(`ard_arm_dim=7`, `state_select`, `action_space="ee"` 등)을 그대로 적용해야 해서
두 기본값 집합이 충돌한다. 대신 이 스크립트는 "프리셋 오버라이드 + 데이터셋에서 얻은
input/output_features"만으로 `SmolVLAConfig`를 만든다(`docs/configurable_ard.md`가
문서화한 사용법과 동일한 패턴).

Colab 제약(세션이 예고 없이 끊김, 백그라운드 실행 불가)에 맞춰 설계된 핵심:
- 10분(기본)마다 Drive에 모델/옵티마이저/스텝/RNG/데이터 순서 상태를 전부 저장한다
  (`checkpoint.py`).
- 다시 실행하면 같은 `--output-dir`(Drive 경로)에서 최신 체크포인트를 자동으로 찾아
  이어한다.
- `--smoke`로 실제 가중치/데이터 없이 전체 파이프라인(GPU 감지 -> 스키마 검증 ->
  정규화 통계 -> 학습 N스텝 -> 체크포인트 재시작 -> 오프라인 평가)이 깨지지 않는지
  50~100스텝으로 빠르게 검증한다.

사용 예:
    python scripts/colab/run_colab_training.py \\
        --drive-root /content/drive/MyDrive/ard_vla_colab \\
        --steps 2000

    # 끊겼다가 다시 접속했을 때: 같은 명령을 그대로 다시 실행하면 자동으로 이어한다.

    # 실제 Hub/Drive 접근 없이 파이프라인 전체를 빠르게 검증:
    python scripts/colab/run_colab_training.py --smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/colab/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/

import batch_source
import checkpoint as ckpt_mod
import dataset_fetch
import env_setup
import episode_split
import eval_offline
import norm_stats
import schema_check
from _smoke_data import build_smoke_batch, build_smoke_eval_samples, build_smoke_features, build_smoke_frames
from _tiny_vlm import FAKE_MODEL_ID, tiny_smolvlm_patch

ROCO_PRESET_PATH = Path(__file__).resolve().parents[2] / "configs" / "presets" / "roco_board.yaml"


def _import_train_ard():
    """`scripts/train_ard.py`를 모듈로 임포트한다(패키지가 아니라 평범한 스크립트라
    경로로 직접 로드) — `split_policy_features`/`warn_if_action_layout_looks_wrong`
    재사용용."""
    path = Path(__file__).resolve().parents[1] / "train_ard.py"
    spec = importlib.util.spec_from_file_location("train_ard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_roco_overrides() -> dict:
    return yaml.safe_load(ROCO_PRESET_PATH.read_text())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-repo-id", default="rocochallenge2025/rocochallenge2026_Industrial_Assembly")
    parser.add_argument("--drive-root", default=None, help="Drive 위 작업 경로(캐시/분할파일/체크포인트/로그). --smoke가 아니면 필수")
    parser.add_argument("--hf-token", default=None, help="docs/colab_free_tier.md 참고 — Colab Secrets로 주는 걸 권장, CLI 인자로 평문 노출 비추천")
    parser.add_argument("--output-dir", default=None, help="생략하면 --drive-root/outputs")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", default="auto", help="정수 또는 'auto'(VRAM 기반 추정, 미검증)")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None, help="생략하면 --split-seed와 동일")
    parser.add_argument("--checkpoint-every-seconds", type=float, default=600.0)
    parser.add_argument("--keep-last-n-checkpoints", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-expert-only", action=argparse.BooleanOptionalAction, default=True, help="VLM 전체 동결(기본 True) — SmolVLAConfig 기존 기본값과 동일")
    parser.add_argument("--load-vlm-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--require-disk-fit", action="store_true", help="Drive 여유 공간이 데이터셋보다 작으면 다운로드 전에 중단")
    parser.add_argument("--smoke", action="store_true", help="실제 Hub/Drive 없이 합성 데이터로 전체 파이프라인만 빠르게 검증")
    parser.add_argument("--smoke-steps", type=int, default=60)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# 학습 객체 구성 (실제 경로/스모크 경로 공유)
# --------------------------------------------------------------------------------------


def build_config(input_features: dict, output_features: dict, roco_overrides: dict, args: argparse.Namespace, device: str, vlm_model_name: str, load_vlm_weights: bool):
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    kwargs = dict(roco_overrides)
    kwargs.update(
        input_features=input_features,
        output_features=output_features,
        device=device,
        vlm_model_name=vlm_model_name,
        load_vlm_weights=load_vlm_weights,
        train_expert_only=args.train_expert_only,
    )
    return SmolVLAConfig(**kwargs)


def build_policy(config) -> "torch.nn.Module":
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy(config)
    policy.to(torch.device(config.device))
    policy.train()
    return policy


def build_processors(config, dataset_stats):
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    return make_smolvla_pre_post_processors(config, dataset_stats=dataset_stats)


def build_optimizer_and_scheduler(config, policy, total_steps: int):
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    scheduler = config.get_scheduler_preset().build(optimizer, total_steps)
    return optimizer, scheduler


def make_scaler(dtype_choice: env_setup.DtypeChoice) -> "torch.cuda.amp.GradScaler | None":
    if dtype_choice.amp_mode == "fp16_scaled" and torch.cuda.is_available():
        return torch.cuda.amp.GradScaler()
    return None


def train_step(policy, preprocessor, batch: dict, optimizer, device: torch.device, dtype_choice: env_setup.DtypeChoice, scaler, grad_clip_norm: float) -> tuple[dict, float]:
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch = preprocessor(batch)
    optimizer.zero_grad()

    if dtype_choice.amp_mode == "bf16":
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss, loss_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
    elif dtype_choice.amp_mode == "fp16_scaled" and scaler is not None:
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            loss, loss_dict = policy.forward(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss, loss_dict = policy.forward(batch)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()

    return loss_dict, float(grad_norm)


def gpu_max_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


# --------------------------------------------------------------------------------------
# 학습 루프 (BatchSource를 통해 실제/스모크 경로가 동일한 코드를 공유)
# --------------------------------------------------------------------------------------


def run_training_loop(
    policy,
    preprocessor,
    optimizer,
    scheduler,
    source: batch_source.BatchSource,
    device: torch.device,
    dtype_choice: env_setup.DtypeChoice,
    scaler,
    grad_clip_norm: float,
    total_steps: int,
    start_step: int,
    checkpoint_dir: Path,
    checkpoint_every_seconds: float,
    keep_last_n_checkpoints: int,
    data_seed: int,
    log_every: int = 20,
) -> dict:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    last_ckpt_time = time.time()
    last_loss = None
    step_times: list[float] = []

    for step in range(start_step + 1, total_steps + 1):
        t0 = time.perf_counter()
        batch = source.next_batch()
        loss_dict, grad_norm = train_step(policy, preprocessor, batch, optimizer, device, dtype_choice, scaler, grad_clip_norm)
        if scheduler is not None:
            scheduler.step()
        step_time = time.perf_counter() - t0
        step_times.append(step_time)
        last_loss = loss_dict["loss"]

        if step == 1 or step % log_every == 0 or step == total_steps:
            mem = gpu_max_memory_gb()
            logging.info(
                "step %d/%d loss=%.4f grad_norm=%.3f step_time=%.3fs gpu_mem_max=%s",
                step, total_steps, last_loss, grad_norm, step_time,
                f"{mem:.2f}GB" if mem is not None else "N/A",
            )

        now = time.time()
        if now - last_ckpt_time >= checkpoint_every_seconds or step == total_steps:
            path = ckpt_mod.checkpoint_path_for_step(checkpoint_dir, step)
            ckpt_mod.save_checkpoint(path, policy, optimizer, step, data_seed, scheduler=scheduler, scaler=scaler, extra={"total_steps": total_steps})
            ckpt_mod.prune_old_checkpoints(checkpoint_dir, keep_last_n_checkpoints)
            logging.info("체크포인트 저장: %s (step=%d)", path, step)
            last_ckpt_time = now

    return {
        "final_step": total_steps,
        "final_loss": last_loss,
        "avg_step_time": sum(step_times) / len(step_times) if step_times else None,
        "max_step_time": max(step_times) if step_times else None,
        "gpu_mem_max_gb": gpu_max_memory_gb(),
    }


# --------------------------------------------------------------------------------------
# 실제 경로
# --------------------------------------------------------------------------------------


def prepare_real_run(args: argparse.Namespace) -> dict:
    """데이터셋을 받고, 스키마를 검증하고, 분할을 만들고, 학습 전용 통계를 계산한다.
    실제 Hub/Drive 접근이 필요해 이 샌드박스에서 끝까지 실행해본 적은 없다(미확인) —
    각 단계는 `dataset_fetch.py`/`schema_check.py`/`episode_split.py`/`norm_stats.py`의
    독립적으로 테스트된 함수만 호출한다."""
    train_ard = _import_train_ard()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    if args.drive_root is None:
        raise ValueError("--smoke가 아니면 --drive-root가 필요합니다(Drive 위 캐시/체크포인트 경로).")
    drive_root = Path(args.drive_root)
    drive_root.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = drive_root / "hf_cache"

    dataset_fetch.fetch_dataset(args.dataset_repo_id, hf_cache_dir, token=args.hf_token)

    # LeRobotDatasetMetadata/LeRobotDataset(이 레포가 vendor로 들고 있는 버전)는 token
    # kwarg를 받지 않는다(확인됨, lerobot_dataset.py의 두 __init__ 시그니처에 token이
    # 없음) — huggingface_hub가 내부적으로 환경변수 HF_TOKEN을 읽으므로, --hf-token이
    # 주어지면 여기서 환경변수로 설정해 이후 모든 Hub 호출(이 두 클래스 포함)에 전파한다.
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    schema_check.validate_schema(ds_meta.info)
    logging.info("스키마 검증 통과: observation.state/action 채널 순서가 기대 순서와 일치합니다.")

    split_path = drive_root / "episode_split.json"
    split = episode_split.load_or_create_split(
        split_path, ds_meta.total_episodes, num_eval=args.num_eval_episodes, seed=args.split_seed
    )
    logging.info("에피소드 분할: 학습 %d개 / 평가 %d개 (seed=%d, 파일=%s)", len(split.train_ids), len(split.eval_ids), args.split_seed, split_path)

    train_stats_dataset = LeRobotDataset(args.dataset_repo_id, root=None, episodes=split.train_ids)
    train_frames_stats = norm_stats.compute_train_stats(train_stats_dataset)

    overrides = load_roco_overrides()
    state_names = (ds_meta.features.get("observation.state") or {}).get("names")
    action_names = (ds_meta.features.get("action") or {}).get("names")
    stats, report = norm_stats.apply_min_std_with_report(
        train_frames_stats, overrides.get("action_min_std"), overrides.get("state_min_std"),
        overrides.get("state_select"), action_names=action_names, state_names=state_names,
    )
    for line in report.summary_lines():
        logging.info(line)

    from lerobot.datasets.utils import dataset_to_policy_features

    full_features = dataset_to_policy_features(ds_meta.features)
    input_features, output_features = train_ard.split_policy_features(full_features)

    from lerobot.datasets.factory import resolve_delta_timestamps

    return dict(
        ds_meta=ds_meta, split=split, stats=stats, min_std_report=report,
        input_features=input_features, output_features=output_features,
        overrides=overrides, drive_root=drive_root,
        resolve_delta_timestamps=resolve_delta_timestamps, LeRobotDataset=LeRobotDataset,
    )


def build_real_batch_source(bundle: dict, config, batch_size: int, data_seed: int, num_workers: int) -> batch_source.BatchSource:
    from torch.utils.data import DataLoader

    delta_timestamps = bundle["resolve_delta_timestamps"](config, bundle["ds_meta"])
    train_dataset = bundle["LeRobotDataset"](
        bundle["ds_meta"].repo_id, episodes=bundle["split"].train_ids, delta_timestamps=delta_timestamps
    )
    steps_per_epoch = len(train_dataset) // batch_size

    def make_iter_for_epoch(epoch: int):
        gen = torch.Generator()
        gen.manual_seed(data_seed + epoch)
        loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=gen, num_workers=num_workers, drop_last=True)
        return iter(loader)

    return batch_source.BatchSource(make_iter_for_epoch, steps_per_epoch)


def build_real_eval_samples(bundle: dict, config) -> list[eval_offline.EvalSample]:
    """평가 에피소드마다 **중간 지점 청크 하나**를 대표 샘플로 쓴다(에피소드당 여러 청크를
    전부 평가하는 게 더 철저하지만, 이 스크립트의 목표는 "베이스라인과 비교해 방향이
    맞는지" 수준의 스모크/베이스라인 신호다). 맨 처음 프레임을 쓰면 "직전 action"이
    존재하지 않아 `prev_action`을 근사해야 하므로, 대신 에피소드 중간 지점을 써서
    `prev_action`(그 직전 실제 프레임의 action)을 진짜 값으로 둔다."""
    from torch.utils.data import default_collate

    delta_timestamps = bundle["resolve_delta_timestamps"](config, bundle["ds_meta"])
    samples = []
    for ep_id in bundle["split"].eval_ids:
        ep_dataset = bundle["LeRobotDataset"](bundle["ds_meta"].repo_id, episodes=[ep_id], delta_timestamps=delta_timestamps)
        mid_idx = max(1, len(ep_dataset) // 2)
        # default_collate([item])로 "batch_size=1 DataLoader가 만들 배치"와 정확히 같은
        # 형태를 만든다 — 직접 dict comprehension으로 unsqueeze하면 텐서가 아닌 필드
        # (예: task 문자열)에서 깨진다.
        collated = default_collate([ep_dataset[mid_idx]])
        target_chunk = collated.pop("action").squeeze(0).clone()
        prev_action = default_collate([ep_dataset[mid_idx - 1]])["action"].squeeze(0)[0].clone()
        samples.append(eval_offline.EvalSample(batch=collated, target_chunk=target_chunk, prev_action=prev_action))
    return samples


# --------------------------------------------------------------------------------------
# 스모크 경로
# --------------------------------------------------------------------------------------


def run_smoke(args: argparse.Namespace) -> dict:
    """실제 Hub/Drive 없이 합성 데이터로 전 구간(GPU 감지/스키마 검증/정규화 통계/
    학습/체크포인트 재시작/오프라인 평가)을 돌려 각 항목의 통과 여부를 보고한다."""
    results: dict[str, dict] = {}

    def record(name: str, ok: bool, detail: str = "") -> None:
        results[name] = {"ok": ok, "detail": detail}
        logging.info("[smoke] %-28s %s %s", name, "PASS" if ok else "FAIL", detail)

    # 1) GPU 감지
    try:
        env = env_setup.detect_compute_env(manual_batch_size=1, force_dtype=None if args.dtype == "auto" else args.dtype)
        record("gpu_detect", True, env.summary().replace("\n", " | "))
    except Exception as e:  # noqa: BLE001 - 스모크는 각 단계를 독립적으로 보고해야 함
        record("gpu_detect", False, str(e))
        return results

    # 2) 스키마 검증 (정상 케이스 + 비정상 케이스 둘 다 확인)
    try:
        raw_names = (
            [f"left_ee_{c}" for c in ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")]
            + [f"right_ee_{c}" for c in ("pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w")]
            + [f"left_joint_pos_{i}" for i in range(7)]
            + [f"right_joint_pos_{i}" for i in range(7)]
            + [f"left_joint_vel_{i}" for i in range(7)]
            + [f"right_joint_vel_{i}" for i in range(7)]
            + ["left_gripper", "right_gripper"]
        )
        action_names = [
            "left_ee_x", "left_ee_y", "left_ee_z", "left_ee_rx", "left_ee_ry", "left_ee_rz", "left_gripper",
            "right_ee_x", "right_ee_y", "right_ee_z", "right_ee_rx", "right_ee_ry", "right_ee_rz", "right_gripper",
        ]
        info_ok = {"features": {"observation.state": {"names": raw_names}, "action": {"names": action_names}}}
        schema_check.validate_schema(info_ok)

        bad_names = list(raw_names)
        bad_names[42], bad_names[43] = bad_names[43], bad_names[42]
        info_bad = {"features": {"observation.state": {"names": bad_names}, "action": {"names": action_names}}}
        raised = False
        try:
            schema_check.validate_schema(info_bad)
        except schema_check.SchemaMismatchError:
            raised = True
        record("schema_check", raised, "정상 통과 + 순서 뒤집은 케이스는 중단됨 확인")
    except Exception as e:  # noqa: BLE001
        record("schema_check", False, str(e))

    # 3) 정규화 통계 + min_std 리포트
    try:
        overrides = load_roco_overrides()
        # state_select가 raw 44차원을 가정하므로(roco_board.yaml), 여기서도 raw 44차원
        # 합성 state를 써야 한다 — _smoke_data의 기본 STATE_DIM(30)은 이미 "선택된 뒤"
        # 크기라 여기 쓰면 state_select 인덱스가 범위를 벗어난다(4)번 학습 루프에서는
        # state_select 자체를 빼므로 30차원을 그대로 쓴다, 서로 다른 테스트 목적).
        frames = build_smoke_frames(200, state_dim=44)
        stats = norm_stats.compute_train_stats(frames)
        _, report = norm_stats.apply_min_std_with_report(
            stats, overrides["action_min_std"], overrides["state_min_std"], overrides["state_select"]
        )
        record("norm_stats", True, f"action_floored={len(report.action_floored)} state_floored={len(report.state_floored)}")
    except Exception as e:  # noqa: BLE001
        record("norm_stats", False, str(e))

    # 4) 학습 루프 + 체크포인트 저장/재시작 (진짜 SmolVLAPolicy, 작은 합성 백본)
    tmp_ckpt_dir = Path(args.output_dir or "outputs/colab_smoke") / "checkpoints"
    tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tiny_smolvlm_patch():
            input_features, output_features = build_smoke_features()
            overrides = dict(load_roco_overrides())
            # state_select는 raw 44차원 observation.state에서 30개를 골라내는 인덱스 리스트인데,
            # 합성 데이터는 애초에 30차원으로(이미 "선택된" 크기로) 만들어서 raw 44차원 자체가
            # 없다 — state_select를 그대로 쓰면 prepare_state()가 범위 밖 인덱스로 크래시한다.
            # state_min_std도 state_select에 종속되므로 같이 뺀다. action_min_std는 차원이
            # 맞긴 하지만(14) 세 옵션을 한 묶음으로 끄는 쪽이 "합성 데이터는 실제 RoCo 채널
            # 분포를 흉내 내지 않는다"는 사실을 더 명확히 드러낸다고 판단해 같이 뺐다.
            overrides.pop("action_min_std", None)
            overrides.pop("state_min_std", None)
            overrides.pop("state_select", None)
            config = build_config(
                input_features, output_features, overrides, args, device="cpu",
                vlm_model_name=FAKE_MODEL_ID, load_vlm_weights=False,
            )
            config.num_vlm_layers = 2
            config.chunk_size = 8
            config.n_action_steps = 8
            config.max_state_dim = 30
            config.max_action_dim = 14

            policy = build_policy(config)
            preprocessor, postprocessor = build_processors(config, dataset_stats=None)
            optimizer, scheduler = build_optimizer_and_scheduler(config, policy, args.smoke_steps)
            dtype_choice = env.dtype_choice
            scaler = make_scaler(dtype_choice)

            pool = [build_smoke_batch(batch_size=2, chunk_size=8) for _ in range(4)]

            def make_iter_for_epoch(_epoch: int):
                return iter(pool)

            source = batch_source.BatchSource(make_iter_for_epoch, steps_per_epoch=len(pool))

            half = args.smoke_steps // 2
            summary1 = run_training_loop(
                policy, preprocessor, optimizer, scheduler, source, torch.device("cpu"), dtype_choice, scaler,
                grad_clip_norm=config.optimizer_grad_clip_norm, total_steps=half, start_step=0,
                checkpoint_dir=tmp_ckpt_dir, checkpoint_every_seconds=0.0, keep_last_n_checkpoints=3,
                data_seed=0, log_every=max(1, half // 3),
            )

            # "세션이 끊김" 재현: 새 policy/optimizer를 만들어서 체크포인트에서 복원.
            policy2 = build_policy(config)
            optimizer2, scheduler2 = build_optimizer_and_scheduler(config, policy2, args.smoke_steps)
            latest = ckpt_mod.find_latest_checkpoint(tmp_ckpt_dir)
            payload = ckpt_mod.load_checkpoint(latest, policy2, optimizer2, scheduler=scheduler2, scaler=scaler)
            resumed_step = payload["step"]
            source2 = batch_source.BatchSource(make_iter_for_epoch, steps_per_epoch=len(pool))
            source2.fast_forward(resumed_step)

            summary2 = run_training_loop(
                policy2, preprocessor, optimizer2, scheduler2, source2, torch.device("cpu"), dtype_choice, scaler,
                grad_clip_norm=config.optimizer_grad_clip_norm, total_steps=args.smoke_steps, start_step=resumed_step,
                checkpoint_dir=tmp_ckpt_dir, checkpoint_every_seconds=0.0, keep_last_n_checkpoints=3,
                data_seed=0, log_every=max(1, (args.smoke_steps - resumed_step) // 3 or 1),
            )
            record(
                "train_and_resume", True,
                f"half_step_loss={summary1['final_loss']:.4f} resumed_at={resumed_step} "
                f"final_loss={summary2['final_loss']:.4f} avg_step_time={summary2['avg_step_time']:.3f}s "
                f"gpu_mem_max={summary2['gpu_mem_max_gb']}",
            )

            # 5) 오프라인 평가
            eval_samples = build_smoke_eval_samples(num_samples=3, chunk_size=config.chunk_size)
            train_mean_action = torch.zeros(14)
            eval_result = eval_offline.run_offline_eval(policy2, preprocessor, postprocessor, eval_samples, train_mean_action, device="cpu")
            report_path = Path(args.output_dir or "outputs/colab_smoke") / "eval_report.json"
            eval_offline.save_eval_report(report_path, eval_result)
            record("offline_eval", True, f"model_overall_l1={eval_result.model['overall']:.4f} report={report_path}")
    except Exception as e:  # noqa: BLE001
        logging.exception("스모크 학습/평가 단계 실패")
        record("train_and_resume_or_eval", False, str(e))

    return results


def print_smoke_summary(results: dict[str, dict]) -> bool:
    print("\n=== --smoke 결과 요약 ===")
    all_ok = True
    for name, r in results.items():
        status = "PASS" if r["ok"] else "FAIL"
        all_ok = all_ok and r["ok"]
        print(f"[{status}] {name}: {r['detail']}")
    print(f"=== 전체: {'PASS' if all_ok else 'FAIL'} ===\n")
    return all_ok


# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)

    if args.smoke:
        results = run_smoke(args)
        ok = print_smoke_summary(results)
        return 0 if ok else 1

    env = env_setup.detect_compute_env(
        manual_batch_size=None if args.batch_size == "auto" else int(args.batch_size),
        force_dtype=None if args.dtype == "auto" else args.dtype,
    )
    logging.info("실행 환경:\n%s", env.summary())

    bundle = prepare_real_run(args)
    config = build_config(
        bundle["input_features"], bundle["output_features"], bundle["overrides"], args,
        device="cuda" if env.gpu.available else "cpu", vlm_model_name=args.vlm_model_name,
        load_vlm_weights=args.load_vlm_weights,
    )
    policy = build_policy(config)
    preprocessor, postprocessor = build_processors(config, dataset_stats=bundle["stats"])
    optimizer, scheduler = build_optimizer_and_scheduler(config, policy, args.steps)
    scaler = make_scaler(env.dtype_choice)

    data_seed = args.data_seed if args.data_seed is not None else args.split_seed
    source = build_real_batch_source(bundle, config, env.batch_size, data_seed, args.num_workers)

    output_dir = Path(args.output_dir) if args.output_dir else bundle["drive_root"] / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    latest = ckpt_mod.find_latest_checkpoint(checkpoint_dir)
    if latest is not None:
        payload = ckpt_mod.load_checkpoint(latest, policy, optimizer, scheduler=scheduler, scaler=scaler)
        start_step = payload["step"]
        source.fast_forward(start_step)
        logging.info("체크포인트에서 재시작: %s (step=%d)", latest, start_step)

    device = torch.device(config.device)
    summary = run_training_loop(
        policy, preprocessor, optimizer, scheduler, source, device, env.dtype_choice, scaler,
        grad_clip_norm=config.optimizer_grad_clip_norm, total_steps=args.steps, start_step=start_step,
        checkpoint_dir=checkpoint_dir, checkpoint_every_seconds=args.checkpoint_every_seconds,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints, data_seed=data_seed, log_every=args.log_every,
    )
    logging.info("학습 종료: %s", summary)

    eval_samples = build_real_eval_samples(bundle, config)
    train_mean_action = bundle["stats"]["action"]["mean"]
    eval_result = eval_offline.run_offline_eval(policy, preprocessor, postprocessor, eval_samples, train_mean_action, device=device)
    report_path = output_dir / "eval_report.json"
    eval_offline.save_eval_report(report_path, eval_result)
    logging.info("오프라인 평가 결과(%s): %s", report_path, json.dumps(eval_result.to_json_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
