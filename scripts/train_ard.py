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

기본적으로 QLoRA 스타일 4bit 양자화(--quantization 기본값 "4bit")로 VLM 백본을 얼려서
로드하고 LoRA 어댑터(--use-lora 기본 켜짐)만 학습한다 — bitsandbytes로 GPU 메모리를
아끼면서 파인튜닝하려는 게 기본 시나리오이기 때문이다. 필요하면 껄 수 있다:
    python scripts/train_ard.py --dataset-repo-id <...> --quantization none --no-lora
`pip install`할 때 `quantization` extra가 필요하다: `pip install -e "third_party/lerobot[quantization]"`.

**실측(Colab GPU, 이 레포에서 확인함) 주의**: 4bit/8bit 양자화는 배치 사이즈가 작을 때만
peak memory를 뚜렷하게 줄인다(배치=1에서 4bit 약 -17~26%). 배치가 커질수록(활성화 메모리가
지배적이 되면서) 절감 효과가 줄어들다가, 8bit은 배치 32에서 오히려 bf16보다 최대 +23% 더
많은 메모리를 쓴다(README의 "QLoRA 스타일 백본 양자화" 절 참고). VRAM이 빠듯해서 배치를
작게 잡아야 하는 상황에서 가장 효과적이고, 배치를 크게 돌릴 수 있는 GPU라면 굳이 켤 이유가
없을 수 있다.
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

    qlora_group = parser.add_argument_group("QLoRA (bitsandbytes 백본 양자화 + LoRA)")
    qlora_group.add_argument(
        "--quantization",
        choices=["none", "4bit", "8bit"],
        default="4bit",
        help=(
            "기본값 '4bit': bitsandbytes로 VLM 백본을 양자화해서 얼리고 LoRA만 학습하는 QLoRA "
            "스타일로 빌드한다. 'none'이면 양자화 없이 bf16 그대로 전체(또는 --use-lora면 "
            "action expert만) 학습한다."
        ),
    )
    qlora_group.add_argument("--bnb-4bit-quant-type", default="nf4", choices=["nf4", "fp4"])
    qlora_group.add_argument(
        "--no-bnb-4bit-double-quant", dest="bnb_4bit_use_double_quant", action="store_false"
    )
    qlora_group.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "기본 켜짐: policy.wrap_with_peft()로 LoRA 어댑터를 붙인다. --quantization이 "
            "'none'이 아니면 이게 꺼져 있으면 얼려진 백본이 아예 학습에 안 낀다(QLoRA 취지에 "
            "안 맞음 — 정상 동작이지만 권장하지 않음)."
        ),
    )
    qlora_group.add_argument("--lora-r", type=int, default=8)
    qlora_group.add_argument("--lora-alpha", type=int, default=16)
    parser.set_defaults(bnb_4bit_use_double_quant=True)

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

    quantization = None if args.quantization == "none" else args.quantization
    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=args.load_vlm_weights,
        # wrap_with_peft()의 "from-scratch 경고"를 통과시키는 용도 — LoRA를 쓸 땐 실제로
        # 사전학습 가중치 위에서 파인튜닝하는 것이므로 사실과 부합한다.
        pretrained_path=args.vlm_model_name if args.use_lora else None,
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
        quantization=quantization,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )
    device = torch.device(config.device)
    logging.info(
        "device=%s use_ard=%s quantization=%s use_lora=%s", device, config.use_ard, quantization, args.use_lora
    )

    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, dataset_stats=dataset.meta.stats)

    policy = SmolVLAPolicy(config)
    policy.to(device)
    policy.train()

    if args.use_lora:
        # peft.get_peft_model()이 대상 서브모듈을 in-place로 교체하므로(peft 표준 동작),
        # 반환값(peft_model)을 따로 안 쓰고 이후에도 `policy` 참조를 그대로 forward/backward에
        # 쓸 수 있다. 양자화가 켜져 있으면 _get_default_peft_targets()가 VLM 백본의 q/v_proj도
        # LoRA 타겟에 자동으로 포함시킨다 (modeling_smolvla.py 참고).
        peft_model = policy.wrap_with_peft(peft_cli_overrides={"r": args.lora_r, "lora_alpha": args.lora_alpha})
        if hasattr(peft_model, "print_trainable_parameters"):
            peft_model.print_trainable_parameters()

        if args.use_ard:
            # 기본 LoRA 타겟에는 ARD의 stabilizer_head/actuator_head가 없어서, wrap_with_peft()가
            # 전체를 얼린 뒤 이 head들도 같이 얼어붙는다 — 다시 학습 가능하게 풀어준다.
            n = 0
            for p in policy.model.ard_heads.parameters():
                p.requires_grad_(True)
                n += p.numel()
            logging.info("LoRA 적용 후 AsymmetricResidualHeads %d개 파라미터를 다시 학습 가능하게 풀었습니다.", n)

    if args.vlm_layer_indices is not None:
        logging.info("VLM 레이어 구성: 사용자 지정 인덱스 %s (n=%d)", sorted(args.vlm_layer_indices), policy.model.vlm_with_expert.num_vlm_layers)
    else:
        logging.info("VLM 레이어 구성: 앞쪽 %d개", policy.model.vlm_with_expert.num_vlm_layers)

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
