#!/usr/bin/env python3
"""SmolVLA(+ARD) 최소 학습 스크립트.

`third_party/lerobot`에서 SmolVLA만 남기고 트림하면서 lerobot의 범용 학습 CLI
(`lerobot_train.py`)도 같이 제거했다 — 그건 모든 정책(ACT/Diffusion/pi0/...)을 알아야 하는
`policies/factory.py`에 의존하고 있어서, 다시 가져오면 트림한 의미가 없어지기 때문이다.
대신 SmolVLA 하나만 알면 되는 이 짧은 스크립트로 실제 학습 루프를 돌린다.

이 레포의 샌드박스는 Hugging Face Hub 접근이 막혀 있어(SmolVLM2 백본 config를 받아야 해서)
여기서는 end-to-end로 실행해볼 수 없다. 실제 GPU가 있는 로컬 환경에서 검증해야 한다.

사용 예:
    python scripts/train_ard.py \
        --dataset-repo-id <HF_USER>/<DATASET> \
        --output-dir outputs/ard_run1 \
        --steps 20000 \
        --batch-size 32

ARD를 끄고 베이스라인 SmolVLA만 학습하려면 --no-use-ard 를 추가한다.

VLM 레이어를 "앞쪽 N개"가 아니라 특정 인덱스 조합으로 구성하려면(예: scripts/layer_importance.py가
코사인 유사도 기준으로 골라준 레이어들) --vlm-layer-indices를 준다:
    python scripts/train_ard.py --dataset-repo-id <...> \
        --vlm-layer-indices 2 3 4 7 8 9 10 11 12 14 15 16 20 21 25 26
먼저 scripts/profile_memory.py --vlm-layer-indices ...로 같은 조합이 메모리/빌드 문제없이
도는지 확인해보는 걸 권장한다.
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import ACTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--dataset-repo-id", required=True, help="LeRobotDataset repo id (예: <user>/<dataset>)")
    parser.add_argument("--output-dir", default="outputs/ard_train", help="체크포인트 저장 경로")
    parser.add_argument("--steps", type=int, default=1000, help="총 학습 스텝 수")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--device", default=None, help="예: cuda, cuda:0, cpu. 지정 안 하면 자동 선택")
    parser.add_argument(
        "--load-vlm-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="SmolVLM2 백본을 사전학습 가중치로 로드할지 여부 (False면 구조만 가져와서 처음부터 학습)",
    )
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument(
        "--num-vlm-layers",
        type=int,
        default=16,
        help="SmolLM2에서 앞쪽 몇 개 레이어만 쓸지 (기본 16, SmolVLAConfig 기본값과 동일). --vlm-layer-indices가 주어지면 무시된다.",
    )
    parser.add_argument(
        "--vlm-layer-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "'앞에서부터 --num-vlm-layers개'라는 기본 규칙 대신, 이 원본 레이어 인덱스 조합을 "
            "그대로 써서 VLM을 구성한다 (예: scripts/layer_importance.py가 코사인 유사도 기준으로 "
            "골라준 레이어들 — scripts/profile_memory.py --vlm-layer-indices ...로 먼저 메모리/빌드"
            "가 정상인지 확인해보는 걸 권장한다). 주어지면 --num-vlm-layers는 무시된다."
        ),
    )

    ard_group = parser.add_argument_group("ARD")
    ard_group.add_argument("--use-ard", action=argparse.BooleanOptionalAction, default=True)
    ard_group.add_argument("--ard-arm-dim", type=int, default=7, help="팔 하나당 자유도 (기본 6관절+그리퍼1)")
    ard_group.add_argument("--ard-actuator-arm", default="right", choices=["left", "right"])
    ard_group.add_argument("--ard-alpha", type=float, default=0.3, help="Stabilizer 손실 가중치")
    ard_group.add_argument("--ard-beta", type=float, default=0.7, help="Actuator 손실 가중치")
    ard_group.add_argument("--ard-lambda-smooth", type=float, default=1.0)
    ard_group.add_argument("--ard-lambda-force", type=float, default=1.0)
    ard_group.add_argument("--ard-lambda-traj", type=float, default=1.0)

    return parser.parse_args()


def split_policy_features(features: dict) -> tuple[dict, dict]:
    """dataset_to_policy_features()의 결과를 SmolVLAConfig가 요구하는
    (input_features, output_features)로 나눈다. ACTION 타입만 output, 나머지(VISUAL/STATE/ENV)는 input."""
    input_features = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
    output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}
    return input_features, output_features


def warn_if_action_layout_looks_wrong(dataset: LeRobotDataset, arm_dim: int) -> None:
    """ARD는 액션 벡터의 앞 arm_dim개 채널=왼팔, 다음 arm_dim개=오른팔이라고 가정한다.
    이 가정이 맞는지 자동으로 검증할 방법은 없으니(dataset은 순서를 보장하지 않는다),
    최소한 사람이 눈으로 확인할 수 있게 액션 채널 이름을 출력해준다."""
    action_feature = dataset.meta.features.get(ACTION)
    if action_feature is None:
        return
    names = action_feature.get("names")
    if not names:
        logging.warning(
            "데이터셋에 액션 채널 이름(names)이 없어서 왼팔/오른팔 순서를 확인할 수 없습니다. "
            "ARD는 앞 %d개=왼팔, 다음 %d개=오른팔이라고 가정하니 직접 확인하세요.",
            arm_dim,
            arm_dim,
        )
        return
    left, right = names[:arm_dim], names[arm_dim : 2 * arm_dim]
    logging.info("액션 채널 순서 (ARD 가정: 앞 %d개=왼팔, 다음 %d개=오른팔):", arm_dim, arm_dim)
    logging.info("  왼팔(Stabilizer 예상)  : %s", left)
    logging.info("  오른팔(Actuator 예상) : %s", right)
    logging.info("  이 순서가 실제 로봇 배선과 다르면 --ard-actuator-arm 이나 데이터셋의 채널 순서를 맞춰야 합니다.")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("데이터셋 로드 중: %s", args.dataset_repo_id)
    dataset = LeRobotDataset(args.dataset_repo_id)

    features = dataset_to_policy_features(dataset.meta.features)
    input_features, output_features = split_policy_features(features)

    if args.use_ard:
        warn_if_action_layout_looks_wrong(dataset, args.ard_arm_dim)

    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=args.load_vlm_weights,
        num_vlm_layers=args.num_vlm_layers,
        vlm_layer_indices=args.vlm_layer_indices,
        use_ard=args.use_ard,
        ard_arm_dim=args.ard_arm_dim,
        ard_default_actuator_arm=args.ard_actuator_arm,
        ard_alpha=args.ard_alpha,
        ard_beta=args.ard_beta,
        ard_lambda_smooth=args.ard_lambda_smooth,
        ard_lambda_force=args.ard_lambda_force,
        ard_lambda_traj=args.ard_lambda_traj,
    )
    device = torch.device(config.device)
    logging.info("device=%s use_ard=%s", device, config.use_ard)

    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, dataset_stats=dataset.meta.stats)

    policy = SmolVLAPolicy(config)
    policy.to(device)
    policy.train()

    if args.vlm_layer_indices is not None:
        logging.info("VLM 레이어 구성: 사용자 지정 인덱스 %s (n=%d)", sorted(args.vlm_layer_indices), policy.model.vlm_with_expert.num_vlm_layers)
    else:
        logging.info("VLM 레이어 구성: 앞쪽 %d개", policy.model.vlm_with_expert.num_vlm_layers)

    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    scheduler = config.get_scheduler_preset().build(optimizer, args.steps)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    data_iter = iter(dataloader)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = preprocessor(batch)
        loss, loss_dict = policy.forward(batch)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.optimizer_grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step == 1 or step % args.log_every == 0:
            msg = f"step {step}/{args.steps}  loss={loss_dict['loss']:.4f}  grad_norm={float(grad_norm):.3f}"
            if config.use_ard:
                msg += (
                    f"  stab={loss_dict['ard_stabilizer_loss']:.4f}"
                    f"  act={loss_dict['ard_actuator_loss']:.4f}"
                    f"  smooth={loss_dict['ard_smooth_loss']:.4f}"
                    f"  force={loss_dict['ard_force_loss']:.4f}"
                    f"  traj={loss_dict['ard_traj_loss']:.4f}"
                )
            logging.info(msg)

        if step % args.save_every == 0 or step == args.steps:
            ckpt_dir = output_dir / f"step_{step:07d}"
            policy.save_pretrained(ckpt_dir)
            preprocessor.save_pretrained(ckpt_dir)
            postprocessor.save_pretrained(ckpt_dir)
            logging.info("체크포인트 저장: %s", ckpt_dir)

    logging.info("학습 완료")


if __name__ == "__main__":
    main()
