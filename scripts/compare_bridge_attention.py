#!/usr/bin/env python3
"""ARD Bridge Attention(VLA-Adapter식)이 실제로 학습 성능(loss)을 개선하는지 직접 비교한다.

scripts/train_ard.py의 학습 루프를 그대로 재사용하되, --use-bridge-attention 켠 모델과 끈
모델을 **같은 데이터 순서**로 순차 학습시켜서(GPU 메모리를 하나만 쓰면서도 공정하게 비교하려고
동시가 아니라 순차로 돈다 — 대신 두 실행 모두 같은 시드로 DataLoader를 만들어서 배치 순서를
동일하게 맞춘다) loss 곡선을 나란히 비교한다.

**중요한 과학적 주의점**: 이건 "파라미터 개수를 맞춘" 통제 실험이 아니다. Bridge Attention을
켜면 ARD head가 학습 가능한 파라미터를 더 갖게 되므로(profile_memory.py로 실측: 약 0.37M ->
약 5.7M, README의 "Bridge Attention" 절 참고), loss가 더 잘 내려간다 해도 그게 "여러 레이어를
조건으로 주는 메커니즘 자체" 덕분인지 "단순히 학습 가능한 파라미터가 더 많아서"인지는 이 비교
만으로는 완전히 분리할 수 없다. 그래도 "구조를 추가했더니 실제로 이 데이터셋에서 loss가
개선되는가/안 되는가"라는 1차 질문에는 바로 답을 준다.

사용 예 (실제 LeRobotDataset repo id 필요):
    python scripts/compare_bridge_attention.py \
        --dataset-repo-id <HF_USER>/<DATASET> \
        --steps 300 \
        --batch-size 8

결과로 (1) 스텝별 loss 비교 표, (2) loss_curve.png(두 곡선 겹쳐 그림), (3) loss_curve.csv
(원본 수치)를 --output-dir(기본 outputs/compare_bridge_attention)에 남긴다.

주의 — 이 스크립트는 GPU/Hugging Face Hub/실제 데이터셋 접근이 모두 없는 샌드박스에서
작성되어 실제로 돌려보지 못했다. train_ard.py와 동일한 학습 루프 구조(옵티마이저 순서,
GradNorm은 이 비교에서는 뺐다 — 비교 변수를 Bridge Attention 하나로 좁히기 위해)를 그대로
따랐고, 두 변형(baseline/bridge)에 동일한 배치 순서를 보장하는 시드 고정 로직은 오프라인에서
합성 데이터로 직접 검증했다(아래 "오프라인 검증" 참고).
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from train_ard import split_policy_features, warn_if_action_layout_looks_wrong  # noqa: E402

from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-repo-id", required=True, help="LeRobotDataset repo id (예: <user>/<dataset>)")
    parser.add_argument("--output-dir", default="outputs/compare_bridge_attention")
    parser.add_argument("--steps", type=int, default=300, help="변형당 학습 스텝 수")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10, help="표/CSV에 몇 스텝마다 기록할지")
    parser.add_argument("--seed", type=int, default=0, help="두 변형이 같은 배치 순서를 보게 만드는 시드")
    parser.add_argument("--device", default=None)

    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--num-vlm-layers", type=int, default=16)
    parser.add_argument("--vlm-layer-indices", type=int, nargs="+", default=None)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false", help="LoRA 끄고(전체 파인튜닝) 비교")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)

    ard_group = parser.add_argument_group("ARD")
    ard_group.add_argument("--ard-arm-dim", type=int, default=7)
    ard_group.add_argument("--ard-actuator-arm", default="right", choices=["left", "right"])
    ard_group.add_argument("--ard-alpha", type=float, default=0.3)
    ard_group.add_argument("--ard-beta", type=float, default=0.7)
    ard_group.add_argument("--ard-bridge-layer-indices", type=int, nargs="+", default=None)
    ard_group.add_argument("--ard-bridge-num-heads", type=int, default=4)

    parser.set_defaults(use_lora=True)
    return parser.parse_args()


def build_config(args, use_bridge_attention: bool, input_features: dict, output_features: dict) -> SmolVLAConfig:
    return SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        device=args.device,
        vlm_model_name=args.vlm_model_name,
        load_vlm_weights=True,
        pretrained_path=args.vlm_model_name if args.use_lora else None,  # wrap_with_peft()의 from-scratch 경고 통과용
        num_vlm_layers=args.num_vlm_layers,
        vlm_layer_indices=args.vlm_layer_indices,
        use_ard=True,
        ard_arm_dim=args.ard_arm_dim,
        ard_default_actuator_arm=args.ard_actuator_arm,
        ard_alpha=args.ard_alpha,
        ard_beta=args.ard_beta,
        use_bridge_attention=use_bridge_attention,
        ard_bridge_layer_indices=args.ard_bridge_layer_indices,
        ard_bridge_num_heads=args.ard_bridge_num_heads,
    )


def apply_lora(policy: SmolVLAPolicy, args) -> None:
    """profile_memory.py의 apply_lora()와 동일한 패턴 — LoRA 적용 후 ARD head(Bridge Attention
    포함)를 다시 학습 가능하게 풀어준다. print_trainable_parameters()는 그 이후에 불러야
    정확한 숫자가 찍힌다(예전에 순서가 바뀌어서 생겼던 버그, profile_memory.py 참고)."""
    peft_model = policy.wrap_with_peft(peft_cli_overrides={"r": args.lora_r, "lora_alpha": args.lora_alpha})
    n = 0
    for p in policy.model.ard_heads.parameters():
        p.requires_grad_(True)
        n += p.numel()
    logging.info("LoRA 적용 후 ARD head(Bridge Attention 포함) %d개 파라미터를 다시 학습 가능하게 풀었습니다.", n)
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()


def run_training(
    label: str,
    args,
    dataset: LeRobotDataset,
    input_features: dict,
    output_features: dict,
    use_bridge_attention: bool,
) -> list[dict]:
    logging.info("=" * 70)
    logging.info("[%s] 학습 시작 (use_bridge_attention=%s)", label, use_bridge_attention)
    logging.info("=" * 70)

    config = build_config(args, use_bridge_attention, input_features, output_features)
    device = torch.device(config.device)
    preprocessor, _ = make_smolvla_pre_post_processors(config, dataset_stats=dataset.meta.stats)

    policy = SmolVLAPolicy(config)
    policy.to(device)
    policy.train()

    if use_bridge_attention:
        logging.info("Bridge Attention 조건 레이어 인덱스(트림된 기준): %s", policy.model.bridge_layer_indices)

    if args.use_lora:
        apply_lora(policy, args)

    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in policy.parameters())
    logging.info("학습 가능 파라미터: %s / 전체 %s (%.3f%%)", f"{n_trainable:,}", f"{n_total:,}", 100 * n_trainable / n_total)

    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    scheduler = config.get_scheduler_preset().build(optimizer, args.steps)

    # 두 변형(baseline/bridge)이 정확히 같은 배치 순서를 보도록, 매번 같은 시드로 새로 만든
    # generator를 DataLoader에 넣는다 — torch.manual_seed()만으로는 DataLoader의 shuffle
    # 순열이 재현되지 않을 수 있어(내부적으로 자기 generator를 쓰므로) 명시적으로 넘긴다.
    generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        drop_last=True, generator=generator,
    )
    data_iter = iter(dataloader)

    records = []
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
        torch.nn.utils.clip_grad_norm_(policy.parameters(), config.optimizer_grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": loss_dict["loss"],
                "ard_stabilizer_loss": loss_dict["ard_stabilizer_loss"],
                "ard_actuator_loss": loss_dict["ard_actuator_loss"],
                "ard_pos_loss": loss_dict["ard_pos_loss"],
            }
            records.append(record)
            logging.info(
                "[%s] step %d/%d  loss=%.4f  stab=%.4f  act=%.4f",
                label, step, args.steps, record["loss"], record["ard_stabilizer_loss"], record["ard_actuator_loss"],
            )

    del policy, optimizer, dataloader
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return records


def summarize(label: str, records: list[dict]) -> float:
    """마지막 20%(최소 1개) 스텝의 평균 loss를 "수렴 loss" 대용값으로 쓴다."""
    n_tail = max(1, len(records) // 5)
    tail = records[-n_tail:]
    mean_loss = sum(r["loss"] for r in tail) / len(tail)
    print(f"[{label}] 마지막 {n_tail}개 기록 평균 loss: {mean_loss:.4f}")
    return mean_loss


def save_csv(path: Path, baseline: list[dict], bridge: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "baseline_loss", "bridge_loss"])
        baseline_by_step = {r["step"]: r["loss"] for r in baseline}
        bridge_by_step = {r["step"]: r["loss"] for r in bridge}
        for step in sorted(set(baseline_by_step) | set(bridge_by_step)):
            writer.writerow([step, baseline_by_step.get(step, ""), bridge_by_step.get(step, "")])
    logging.info("CSV 저장: %s", path)


def save_plot(path: Path, baseline: list[dict], bridge: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([r["step"] for r in baseline], [r["loss"] for r in baseline], label="baseline (bridge attention off)", color="#55636F", linewidth=1.6)
    ax.plot([r["step"] for r in bridge], [r["loss"] for r in bridge], label="use_bridge_attention=True", color="#1E7B78", linewidth=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("ARD Bridge Attention: training loss comparison")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logging.info("그래프 저장: %s", path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("데이터셋 메타데이터 로드 중: %s", args.dataset_repo_id)
    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    features = dataset_to_policy_features(ds_meta.features)
    input_features, output_features = split_policy_features(features)

    # train_ard.py와 동일한 이유로 delta_timestamps가 필요하다 — 없으면 action이 단일
    # 프레임으로만 나와서 embed_suffix()가 기대하는 (B, chunk_size, action_dim) 형태가
    # 깨진다(자세한 설명은 train_ard.py의 해당 주석 참고). baseline/bridge 두 변형 모두
    # chunk_size(SmolVLAConfig 기본값)가 같으므로, 대표로 하나의 config만 만들어 쓴다.
    probe_config = build_config(args, use_bridge_attention=False, input_features=input_features, output_features=output_features)
    delta_timestamps = resolve_delta_timestamps(probe_config, ds_meta)

    logging.info("데이터셋 로드 중: %s", args.dataset_repo_id)
    dataset = LeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_timestamps)
    warn_if_action_layout_looks_wrong(dataset, args.ard_arm_dim)

    torch.manual_seed(args.seed)
    baseline_records = run_training("baseline", args, dataset, input_features, output_features, use_bridge_attention=False)

    torch.manual_seed(args.seed)
    bridge_records = run_training("bridge", args, dataset, input_features, output_features, use_bridge_attention=True)

    print("\n" + "=" * 70)
    print(f"{'step':>6} | {'baseline loss':>14} | {'bridge loss':>12} | {'차이(bridge-baseline)':>20}")
    print("-" * 70)
    baseline_by_step = {r["step"]: r["loss"] for r in baseline_records}
    bridge_by_step = {r["step"]: r["loss"] for r in bridge_records}
    for step in sorted(set(baseline_by_step) & set(bridge_by_step)):
        b, g = baseline_by_step[step], bridge_by_step[step]
        print(f"{step:>6} | {b:>14.4f} | {g:>12.4f} | {g - b:>+20.4f}")

    print()
    baseline_tail = summarize("baseline", baseline_records)
    bridge_tail = summarize("bridge", bridge_records)
    diff_pct = 100 * (bridge_tail - baseline_tail) / baseline_tail
    if bridge_tail < baseline_tail:
        print(f"\n-> bridge attention 쪽이 마지막 구간 평균 loss가 {abs(diff_pct):.1f}% 더 낮습니다 (개선).")
    else:
        print(f"\n-> bridge attention 쪽이 마지막 구간 평균 loss가 {abs(diff_pct):.1f}% 더 높습니다 (개선 없음/악화).")
    print(
        "참고: Bridge Attention은 ARD head의 학습 가능 파라미터 자체를 늘리므로(README 'Bridge "
        "Attention' 절 참고), 이 비교만으로 \"메커니즘 자체의 효과\"와 \"파라미터 증가 효과\"를 "
        "완전히 분리할 수는 없습니다 — 이 데이터셋에서 실제로 도움이 되는지/안 되는지의 1차 "
        "판단 용도로 쓰세요."
    )

    output_dir = Path(args.output_dir)
    save_csv(output_dir / "loss_curve.csv", baseline_records, bridge_records)
    save_plot(output_dir / "loss_curve.png", baseline_records, bridge_records)


if __name__ == "__main__":
    main()
