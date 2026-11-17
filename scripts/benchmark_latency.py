#!/usr/bin/env python3
"""SmolVLA(+ARD) select_action() 지연/메모리/파라미터 수를 측정한다.

RoCo 프리셋(`configs/presets/roco_board.yaml`) 기준 배치 1 추론 성능을 본다. `policy_interface.py`
(이 레포의 RoCo 평가 어댑터)를 그대로 재사용한다 — 이미지 dtype 변환/전처리-forward-후처리를
한 번에 묶는 `RoCoPolicyAdapter.act()`를 다시 구현하지 않는다(ponytail: 이미 있는 걸 또 만들지
않는다).

**"청크당 지연" 측정 방법**: 매 측정 반복 전에 `adapter.reset()`을 불러서 액션 큐를 비운다 —
이러면 `select_action()` 호출마다 큐에서 꺼내 쓰는 게 아니라 **매번 새로 forward**해서 청크
전체(`chunk_size`개 액션)를 다시 계산하므로, "큐 pop 지연"이 아니라 "청크 하나를 만드는 데
걸리는 지연"을 그대로 잰다.

사용 예:
    python scripts/benchmark_latency.py --random-init
    python scripts/benchmark_latency.py --checkpoint-path outputs/ard_train/step_0001000

주의 — 이 레포 샌드박스는 GPU/Hugging Face Hub 접근이 둘 다 막혀 있어 실제로 실행해보지
못했다(미검증). `--random-init`도 `--vlm-model-name`의 구조(config)는 Hub에서 받아야 하므로
(`load_vlm_weights=False`라 가중치는 안 받음, count_params.py와 동일한 전제) 로컬/Colab처럼
Hub 접근이 되는 환경에서 돌려야 한다. `tests/test_benchmark_latency.py`는 tiny 합성 백본
(`tiny_smolvlm_patch` 패턴)으로 로직만(지연 측정/JSON 저장/파라미터 집계) CPU에서 검증했다 —
실제 GPU 피크 메모리 측정 자체는 미검증.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/ (policy_interface 임포트용)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "lerobot" / "src"))

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

from policy_interface import ACTION_DIM, DEFAULT_CAMERA_KEYS, STATE_DIM, RoCoPolicyAdapter  # noqa: E402

ROCO_PRESET_PATH = Path(__file__).resolve().parents[1] / "configs" / "presets" / "roco_board.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", default=None, help="SmolVLAPolicy.from_pretrained()로 불러올 저장 경로")
    parser.add_argument(
        "--random-init", action="store_true",
        help="체크포인트 없이 --preset 설정대로 무작위 초기화한 정책을 빌드한다. --checkpoint-path와 정확히 하나만 줘야 한다.",
    )
    parser.add_argument("--preset", default=str(ROCO_PRESET_PATH), help="--random-init일 때만 적용되는 SmolVLAConfig override YAML")
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct", help="--random-init일 때만 사용(구조만 Hub에서 받음, load_vlm_weights=False)")
    parser.add_argument("--num-vlm-layers", type=int, default=16, help="--random-init일 때만 사용(SmolVLAConfig 기본값과 동일)")
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--task", default="insert the bolt into the board", help="지연에 토큰화 비용까지 포함시키기 위한 더미 지시문")
    parser.add_argument(
        "--denoise-steps", type=int, nargs="+", default=[1, 2, 5, 10],
        help="policy.config.num_steps(flow-matching denoise 반복 횟수)를 이 값들로 바꿔가며 각각 측정한다",
    )
    parser.add_argument("--num-samples", type=int, default=20, help="denoise-steps 값마다 측정할 반복 횟수(warmup 제외)")
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--device", default=None, help="생략하면 CUDA 가능하면 cuda, 아니면 cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", default="outputs/benchmark_latency.json")
    return parser.parse_args()


def load_preset_overrides(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def build_policy(args: argparse.Namespace, device: torch.device) -> SmolVLAPolicy:
    if args.random_init:
        input_features = {
            key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, args.image_height, args.image_width))
            for key in DEFAULT_CAMERA_KEYS.values()
        }
        input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))
        output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
        overrides = load_preset_overrides(args.preset)
        config = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            device=str(device),
            vlm_model_name=args.vlm_model_name,
            load_vlm_weights=False,
            num_vlm_layers=args.num_vlm_layers,
            max_state_dim=STATE_DIM,
            max_action_dim=ACTION_DIM,
            **overrides,
        )
        policy = SmolVLAPolicy(config)
    else:
        policy = SmolVLAPolicy.from_pretrained(args.checkpoint_path)
    policy.to(device)
    policy.eval()
    return policy


def count_params(policy: SmolVLAPolicy) -> tuple[int, int]:
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return total, trainable


def measure_latency_for_num_steps(
    adapter: RoCoPolicyAdapter,
    images: dict[str, np.ndarray],
    state: np.ndarray,
    task: str,
    num_steps: int,
    num_warmup: int,
    num_samples: int,
    device: torch.device,
) -> dict:
    adapter.policy.config.num_steps = num_steps

    for _ in range(num_warmup):
        adapter.reset()
        adapter.act(images=images, state=state, task=task)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    samples_ms: list[float] = []
    for _ in range(num_samples):
        adapter.reset()  # 큐를 비워서 이번 act()가 반드시 새로 forward하게 만든다
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        adapter.act(images=images, state=state, task=task)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    peak_gpu_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else None

    return {
        "num_denoise_steps": num_steps,
        "latency_ms_median": float(np.median(samples_ms)),
        "latency_ms_p95": float(np.percentile(samples_ms, 95)),
        "latency_ms_samples": samples_ms,
        "peak_gpu_mem_gb": peak_gpu_mem_gb,
    }


def run_benchmark(args: argparse.Namespace) -> dict:
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    policy = build_policy(args, device)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(policy.config, dataset_stats=None)
    adapter = RoCoPolicyAdapter(policy, preprocessor, postprocessor)

    total_params, trainable_params = count_params(policy)

    rng = np.random.default_rng(args.seed)
    images = {
        role: rng.integers(0, 256, size=(args.image_height, args.image_width, 3), dtype=np.uint8)
        for role in DEFAULT_CAMERA_KEYS
    }
    state = rng.normal(size=(STATE_DIM,)).astype(np.float32)

    sweep = [
        measure_latency_for_num_steps(adapter, images, state, args.task, n, args.num_warmup, args.num_samples, device)
        for n in args.denoise_steps
    ]

    return {
        "device": str(device),
        "checkpoint_path": args.checkpoint_path,
        "random_init": args.random_init,
        "preset": args.preset if args.random_init else None,
        "batch_size": 1,
        "chunk_size": policy.config.chunk_size,
        "n_action_steps": policy.config.n_action_steps,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "denoise_step_sweep": sweep,
    }


def print_report(report: dict) -> None:
    print(f"device={report['device']}  chunk_size={report['chunk_size']}  n_action_steps={report['n_action_steps']}")
    print(f"파라미터: 전체 {report['total_params']:,} / 학습 가능 {report['trainable_params']:,}")
    print(f"\n{'num_steps':>10} | {'median(ms)':>11} | {'p95(ms)':>9} | {'peak mem(GB)':>13}")
    print("-" * 54)
    for r in report["denoise_step_sweep"]:
        mem = f"{r['peak_gpu_mem_gb']:.3f}" if r["peak_gpu_mem_gb"] is not None else "N/A(CPU)"
        print(f"{r['num_denoise_steps']:>10} | {r['latency_ms_median']:>11.2f} | {r['latency_ms_p95']:>9.2f} | {mem:>13}")


def main() -> int:
    args = parse_args()
    if bool(args.checkpoint_path) == bool(args.random_init):
        raise SystemExit("--checkpoint-path 또는 --random-init 중 정확히 하나를 줘야 합니다.")

    report = run_benchmark(args)
    print_report(report)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n결과 저장: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
