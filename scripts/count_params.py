#!/usr/bin/env python3
"""SmolVLA(+ARD) 파라미터를 구성 요소별로 집계하고, 해상도별 이미지 토큰 수를 실측한다.

비전 인코더/LLM/Action Expert/ARD head를 통째로 구성하는 SmolVLMWithExpertModel을
load_vlm_weights=False로 만든다 — 가중치 전체를 받지 않고 구조(config)만 받으므로 훨씬
가볍지만, 그래도 Hugging Face Hub에서 config는 받아야 한다. 이 레포의 샌드박스는 Hub 접근이
막혀 있어서 여기서는 실행할 수 없다 — 로컬 GPU(혹은 CPU도 가능, 가중치 없이 구조만 만드는
거라 느리지 않음) 환경에서 돌려서 실제 숫자를 확인해야 한다.

사용 예:
    python scripts/count_params.py
    python scripts/count_params.py --resolutions 384 512 768
"""

import argparse
from collections import OrderedDict

import torch

from lerobot.policies.smolvla.ard import AsymmetricResidualHeads
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vlm-model-name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--num-vlm-layers", type=int, default=16, help="SmolVLAConfig 기본값과 동일")
    parser.add_argument("--expert-width-multiplier", type=float, default=0.75, help="SmolVLAConfig 기본값과 동일")
    parser.add_argument("--ard-arm-dim", type=int, default=7)
    parser.add_argument("--max-state-dim", type=int, default=32, help="SmolVLAConfig 기본값과 동일")
    parser.add_argument("--max-action-dim", type=int, default=32, help="SmolVLAConfig 기본값과 동일")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[384, 512],
        help="이미지 패치/토큰 수를 확인할 정사각형 해상도 목록 (SmolVLAConfig 기본 resize는 512x512)",
    )
    args = parser.parse_args()

    print(f"모델: {args.vlm_model_name}  (load_vlm_weights=False, 구조만 로드 — 실제 가중치는 다운로드 안 함)")
    print(f"num_vlm_layers={args.num_vlm_layers}  expert_width_multiplier={args.expert_width_multiplier}\n")

    vlm_expert = SmolVLMWithExpertModel(
        model_id=args.vlm_model_name,
        load_vlm_weights=False,
        num_vlm_layers=args.num_vlm_layers,
        expert_width_multiplier=args.expert_width_multiplier,
        device="cpu",
    )

    vision_model = vlm_expert.get_vlm_model().vision_model
    connector = vlm_expert.get_vlm_model().connector
    text_model = vlm_expert.get_vlm_model().text_model
    lm_expert = vlm_expert.lm_expert

    ard_heads = AsymmetricResidualHeads(expert_hidden_size=vlm_expert.expert_hidden_size, arm_dim=args.ard_arm_dim)

    # SmolVLAConfig가 VLAFlowMatching.__init__에서 추가로 만드는 나머지 shim 레이어들
    # (state_proj / action_in_proj / action_out_proj / action_time_mlp_in / action_time_mlp_out)
    text_hidden = text_model.config.hidden_size
    shim_layers = [
        torch.nn.Linear(args.max_state_dim, text_hidden),
        torch.nn.Linear(args.max_action_dim, vlm_expert.expert_hidden_size),
        torch.nn.Linear(vlm_expert.expert_hidden_size, args.max_action_dim),
        torch.nn.Linear(vlm_expert.expert_hidden_size * 2, vlm_expert.expert_hidden_size),
        torch.nn.Linear(vlm_expert.expert_hidden_size, vlm_expert.expert_hidden_size),
    ]

    original_num_layers = vlm_expert.config.text_config.num_hidden_layers

    components = OrderedDict(
        [
            ("비전 인코더 (vision_model)", count_params(vision_model)),
            ("커넥터 (connector, modality projection)", count_params(connector)),
            (
                f"LLM 텍스트 트랜스포머 (text_model, {vlm_expert.num_vlm_layers}/{original_num_layers} 레이어만 사용)",
                count_params(text_model),
            ),
            (
                f"Action Expert (lm_expert, {vlm_expert.num_expert_layers}레이어, hidden={vlm_expert.expert_hidden_size})",
                count_params(lm_expert),
            ),
            ("projection/시간 임베딩 shim 레이어", sum(count_params(m) for m in shim_layers)),
            (f"AsymmetricResidualHeads (ARD, arm_dim={args.ard_arm_dim})", count_params(ard_heads)),
        ]
    )
    total = sum(components.values())

    print(f"{'구성 요소':60s} {'파라미터 수':>14s} {'비중':>8s}")
    print("-" * 88)
    for name, n in components.items():
        print(f"{name:60s} {n:14,d} {n / total * 100:7.2f}%")
    print("-" * 88)
    print(f"{'합계':60s} {total:14,d} {100.0:7.2f}%")

    print(
        f"\nLLM 요약: 전체 {original_num_layers}개 레이어 중 {vlm_expert.num_vlm_layers}개만 사용 | "
        f"hidden_size={text_hidden} | "
        f"num_attention_heads={vlm_expert.num_attention_heads} | "
        f"num_key_value_heads={vlm_expert.num_key_value_heads}"
    )

    # 3분류 요약: SigLIP 비전 인코더 / SmolLM2 언어모델 / action expert(액션 헤드).
    # 커넥터(vision->LLM projection)는 비전 쪽에, shim projection들과 ARD head는 액션 쪽에 묶었다.
    vision_total = count_params(vision_model) + count_params(connector)
    llm_total = count_params(text_model)
    action_total = count_params(lm_expert) + sum(count_params(m) for m in shim_layers) + count_params(ard_heads)
    grand_total = vision_total + llm_total + action_total

    print(f"\n{'3분류 요약':60s} {'파라미터 수':>14s} {'비중':>8s}")
    print("-" * 88)
    print(f"{'SigLIP 비전 인코더 (vision_model + connector)':60s} {vision_total:14,d} {vision_total / grand_total * 100:7.2f}%")
    print(f"{'SmolLM2 언어모델 (text_model)':60s} {llm_total:14,d} {llm_total / grand_total * 100:7.2f}%")
    print(
        f"{'action expert (lm_expert + projection shim + ARD head)':60s} "
        f"{action_total:14,d} {action_total / grand_total * 100:7.2f}%"
    )
    print("-" * 88)
    print(f"{'합계':60s} {grand_total:14,d} {100.0:7.2f}%")

    print("\n해상도별 이미지 패치/토큰 수 (vision_model + connector 통과 후, 실측):")
    with torch.no_grad():
        for res in args.resolutions:
            dummy = torch.zeros(1, 3, res, res)
            emb = vlm_expert.embed_image(dummy)
            print(f"  {res}x{res} -> 토큰 {emb.shape[1]}개 (embedding dim {emb.shape[2]})")


if __name__ == "__main__":
    main()
