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

lambda_smooth/force/traj를 고정값(기본 1.0) 대신 GradNorm(Chen et al., 2018)으로 매 스텝
자동 조정하려면 --use-gradnorm을 준다:
    python scripts/train_ard.py --dataset-repo-id <...> --use-gradnorm --gradnorm-alpha 1.5
자세한 구현은 lerobot.policies.smolvla.ard.GradNormLambdas 참고. --ard-lambda-* 값은 이
모드에서는 무시된다(항상 1.0에서 시작).

ARD head가 suffix_out(action expert의 마지막 지점 출력) 하나만이 아니라 SmolLM2 백본의 여러
중간 레이어 특징까지 cross-attention으로 조건받게 하려면(VLA-Adapter, Wang et al., 2025식
Bridge Attention) --use-bridge-attention을 준다:
    python scripts/train_ard.py --dataset-repo-id <...> --use-bridge-attention
자세한 구현은 lerobot.policies.smolvla.ard.BridgeAttention 참고. 기본은 꺼져 있고(기존과
완전히 동일하게 동작), 켜면 --ard-bridge-layer-indices로 조건에 쓸 레이어를 직접 고르거나
(기본은 [num_vlm_layers의 1/4, 1/2, 3/4, 마지막] 4개 지점 자동 선택), --ard-bridge-num-heads로
cross-attention 헤드 수를 조정할 수 있다.

기존 flow-matching 회귀 손실에 FreqPolicy(2025)식 주파수 영역 일관성 손실을 추가하려면
--use-freq-policy를 준다 (SmolVLA의 병렬 flow-matching 디코딩 자체는 그대로 유지하고, 그
논문의 "저주파[궤적 전체 형태] 우선" 통찰만 additive loss로 가져온 것 — 자세한 설계 의도는
lerobot.policies.smolvla.freq_policy 참고):
    python scripts/train_ard.py --dataset-repo-id <...> --use-freq-policy
--use-ard와 무관하게(둘 다 꺼도) 독립적으로 켤 수 있다. --freq-lambda로 total loss에 더해지는
가중치를, --freq-decay로 저주파 우선 정도를 조정한다(0이면 시간 영역 MSE와 동일, 클수록 저주파
쪽으로 쏠림).
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
    ard_group.add_argument("--ard-lambda-smooth", type=float, default=1.0, help="--use-gradnorm이면 무시됨 (GradNorm은 항상 1.0에서 시작)")
    ard_group.add_argument("--ard-lambda-force", type=float, default=1.0, help="--use-gradnorm이면 무시됨 (GradNorm은 항상 1.0에서 시작)")
    ard_group.add_argument("--ard-lambda-traj", type=float, default=1.0, help="--use-gradnorm이면 무시됨 (GradNorm은 항상 1.0에서 시작)")
    ard_group.add_argument(
        "--use-gradnorm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="lambda_smooth/force/traj를 고정값 대신 GradNorm(Chen et al., 2018)으로 매 스텝 자동 조정한다.",
    )
    ard_group.add_argument("--gradnorm-alpha", type=float, default=1.5, help="GradNorm asymmetry 하이퍼파라미터 (논문 기본값)")
    ard_group.add_argument("--gradnorm-lr", type=float, default=0.025, help="lambda 전용 옵티마이저 학습률")
    ard_group.add_argument(
        "--use-bridge-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "VLA-Adapter(Wang et al., 2025)식 Bridge Attention — ARD head가 suffix_out 하나만이 "
            "아니라 SmolLM2 백본의 여러 중간 레이어 특징도 cross-attention으로 조건받는다. "
            "기본은 꺼져 있어 기존과 완전히 동일하게 동작한다(비교 실험용 플래그)."
        ),
    )
    ard_group.add_argument(
        "--ard-bridge-layer-indices",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Bridge Attention이 조건으로 쓸 VLM 레이어 인덱스(트림된 num_vlm_layers 기준, 원본 "
            "32개 기준이 아님). 안 주면 [1/4, 1/2, 3/4, 마지막] 4개 지점을 자동으로 고른다."
        ),
    )
    ard_group.add_argument("--ard-bridge-num-heads", type=int, default=4, help="Bridge Attention cross-attention 헤드 수")

    freq_group = parser.add_argument_group("FreqPolicy")
    freq_group.add_argument(
        "--use-freq-policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "FreqPolicy(2025)식 주파수 영역 일관성 손실을 기존 flow-matching 회귀 손실에 추가한다 "
            "— 생성 메커니즘(병렬 flow-matching 디코딩) 자체는 그대로 두고, '저주파(궤적 전체 "
            "형태)가 고주파(디테일)보다 우선'이라는 통찰만 additive loss로 가져온 것이다. "
            "--use-ard와 무관하게 독립적으로 켤 수 있다."
        ),
    )
    freq_group.add_argument("--freq-lambda", type=float, default=1.0, help="주파수 일관성 손실을 total loss에 더할 때의 가중치")
    freq_group.add_argument(
        "--freq-decay",
        type=float,
        default=2.0,
        help="0이면 시간 영역 MSE와 수학적으로 동일(Parseval 정리). 클수록 저주파에 더 쏠린 가중치 (chunk_size와 무관하게 정규화됨).",
    )

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
        use_gradnorm=args.use_gradnorm,
        gradnorm_alpha=args.gradnorm_alpha,
        gradnorm_lr=args.gradnorm_lr,
        use_bridge_attention=args.use_bridge_attention,
        ard_bridge_layer_indices=args.ard_bridge_layer_indices,
        ard_bridge_num_heads=args.ard_bridge_num_heads,
        use_freq_policy=args.use_freq_policy,
        freq_lambda=args.freq_lambda,
        freq_decay=args.freq_decay,
    )
    device = torch.device(config.device)
    logging.info(
        "device=%s use_ard=%s use_bridge_attention=%s use_freq_policy=%s",
        device, config.use_ard, config.use_bridge_attention, config.use_freq_policy,
    )

    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, dataset_stats=dataset.meta.stats)

    policy = SmolVLAPolicy(config)
    policy.to(device)
    policy.train()

    if args.vlm_layer_indices is not None:
        logging.info("VLM 레이어 구성: 사용자 지정 인덱스 %s (n=%d)", sorted(args.vlm_layer_indices), policy.model.vlm_with_expert.num_vlm_layers)
    else:
        logging.info("VLM 레이어 구성: 앞쪽 %d개", policy.model.vlm_with_expert.num_vlm_layers)

    if config.use_bridge_attention:
        logging.info("Bridge Attention 조건 레이어 인덱스(트림된 기준): %s", policy.model.bridge_layer_indices)

    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    scheduler = config.get_scheduler_preset().build(optimizer, args.steps)

    gradnorm_optimizer = None
    if config.use_gradnorm:
        # 반드시 policy.to(device) 이후에 만든다 — GradNormLambdas는 일부러 자기 옵티마이저를
        # 갖지 않는다(모듈 생성 시점에 옵티마이저를 만들면 .to(device)가 파라미터 텐서를
        # 바꿔치기했을 때 옵티마이저가 옛 텐서를 참조하는 채로 남는 문제가 생긴다).
        gradnorm_optimizer = torch.optim.Adam([policy.model.ard_gradnorm.weights], lr=config.gradnorm_lr)
        logging.info(
            "GradNorm 활성화: alpha=%.2f lr=%.4f (lambda_smooth/force/traj 전부 1.0에서 시작해 자동 조정됨 — "
            "--ard-lambda-* 값은 무시됨)",
            config.gradnorm_alpha, config.gradnorm_lr,
        )

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

        grad_loss = loss_dict.pop("ard_grad_loss_tensor", None)  # 살아있는 텐서라 로깅 전에 빼둔다

        # 순서가 중요하다: GradNorm의 grad_loss.backward()와 메인 loss.backward()를 *둘 다*
        # 먼저 끝낸 뒤에야 옵티마이저 step을 밟는다. gradnorm_optimizer.step()/renormalize()가
        # lambda 파라미터를 in-place로 바꾸는데, 그게 먼저 일어나면 메인 loss의 그래프가
        # (그 안에서 lambda를 detach해서 참조하고 있으므로) "in-place로 바뀐 값" 에러를 낸다.
        if gradnorm_optimizer is not None and grad_loss is not None:
            gradnorm_optimizer.zero_grad()
            grad_loss.backward(inputs=[policy.model.ard_gradnorm.weights], retain_graph=True)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.optimizer_grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if gradnorm_optimizer is not None and grad_loss is not None:
            gradnorm_optimizer.step()
            policy.model.ard_gradnorm.renormalize()

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
            if config.use_gradnorm:
                msg += (
                    f"  grad_loss={loss_dict['ard_grad_loss']:.4f}"
                    f"  lambda(s/f/t)={loss_dict['ard_lambda_smooth']:.3f}/"
                    f"{loss_dict['ard_lambda_force']:.3f}/{loss_dict['ard_lambda_traj']:.3f}"
                )
            if config.use_freq_policy:
                msg += f"  freq_consistency={loss_dict['freq_consistency_loss']:.4f}"
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
