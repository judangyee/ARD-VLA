#!/usr/bin/env python3
"""LAC(Latent-Aware Controller) 세 모드(off/covla/role_aware)를 작은 합성 SmolVLM 백본 +
몽키패치(`tests/conftest.py`와 같은 패턴, Hugging Face Hub 접근 없음)로 몇 "에피소드" 돌려서
ρ/ω/α 분포와 regime 비율, 액션 떨림(jerk) 지표를 모드별로 비교한다. 실제 환경 루프는 없고
(이 레포에 아직 실제 OpenArm 시뮬 연동이 없음), `select_action()`을 반복 호출하는 것으로
"에피소드"를 흉내 낸다.

**중요한 주의사항** — `AsymmetricResidualHeads`(ARD 헤드)는 학습 초반 항등(identity)으로
시작하도록 마지막 레이어가 **0으로 초기화**되어 있다(`ard.py` 참고). 즉 학습 전(이 스크립트가
만드는 합성 모델은 전혀 학습하지 않음) 잔차(residual)는 항상 정확히 0이고, 그래서 ρ=0이
항상 `tau_rho`보다 작아 **매 스텝이 결정론적으로 "macro" regime으로 분류된다** — regime
분포 자체는 이 상태로는 아무 의미가 없다. 이 스크립트는 `--perturb-ard-heads`(기본 켬)로
ARD 헤드의 마지막 레이어에 작은 가우시안 잡음을 주입해서 "약간 학습된" 것처럼 흉내 낸
잔차를 만들어 세 regime이 실제로 갈리는 모습을 보여준다 — **이것도 실제 학습된 가중치가
아니라 데모용 인위적 섭동이다.** 실제 SmolVLM2 가중치나 OpenArm 시뮬에서는 아직 전혀
돌려보지 않았다(README/docs의 "검증됨 vs 미검증" 참고).
"""

from __future__ import annotations

import argparse
import logging
import statistics

import torch
import transformers
from transformers import SmolVLMConfig, SmolVLMVisionConfig

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

FAKE_MODEL_ID = "fake/tiny-smolvlm"


class _StubTokenizer:
    fake_image_token_id = 1
    global_image_token_id = 2


class _StubProcessor:
    tokenizer = _StubTokenizer()


def _build_tiny_smolvlm_config() -> SmolVLMConfig:
    """`tests/conftest.py`의 합성 SmolVLM config와 동일 — CPU에서 몇 초 안에 forward가 끝난다."""
    vision_config = SmolVLMVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_channels=3, image_size=64, patch_size=16,
    )
    return SmolVLMConfig(
        vision_config=vision_config,
        text_config={
            "model_type": "llama", "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
            "vocab_size": 256, "max_position_embeddings": 128,
        },
        scale_factor=2, image_token_id=57,
    )


def _apply_monkeypatch(tiny_config: SmolVLMConfig) -> None:
    transformers.AutoConfig.from_pretrained = staticmethod(lambda *a, **k: tiny_config)
    transformers.AutoProcessor.from_pretrained = staticmethod(lambda *a, **k: _StubProcessor())


def build_policy(arm_dim: int, chunk_size: int, lac_mode: str, perturb_ard_heads: bool) -> SmolVLAPolicy:
    input_features = {
        OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2 * arm_dim,)),
    }
    output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2 * arm_dim,))}
    config = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
        vlm_model_name=FAKE_MODEL_ID,
        load_vlm_weights=False,
        num_vlm_layers=2,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        max_state_dim=2 * arm_dim,
        max_action_dim=2 * arm_dim,
        use_ard=True,
        ard_arm_dim=arm_dim,
        num_steps=2,  # denoise 스텝 수 — 비교용이라 속도를 위해 최소로
        lac_mode=lac_mode,
        lac_log=lac_mode != "none",
    )
    policy = SmolVLAPolicy(config)
    policy.eval()

    if perturb_ard_heads:
        # 데모용 인위적 섭동 — "약간 학습된" 것처럼 잔차가 0이 아니게 만든다(스크립트 docstring
        # 참고, 실제 학습된 가중치가 아님).
        gen = torch.Generator().manual_seed(0)
        for head in (policy.model.ard_heads.stabilizer_head, policy.model.ard_heads.actuator_head):
            last_linear = head[-1]
            last_linear.weight.data += 0.05 * torch.randn(last_linear.weight.shape, generator=gen)
            last_linear.bias.data += 0.05 * torch.randn(last_linear.bias.shape, generator=gen)

    return policy


def build_batch(batch_size: int, arm_dim: int, lang_len: int = 6) -> dict:
    return {
        OBS_IMAGE: torch.rand(batch_size, 3, 64, 64),
        OBS_STATE: torch.randn(batch_size, 2 * arm_dim),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 200, (batch_size, lang_len)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, lang_len, dtype=torch.bool),
    }


def _jerk(actions: list[torch.Tensor]) -> float:
    """3차 차분(jerk) 크기의 평균 — 값이 작을수록 액션이 덜 떨린다."""
    if len(actions) < 4:
        return float("nan")
    stacked = torch.stack(actions)  # (steps, batch, action_dim)
    d1 = stacked[1:] - stacked[:-1]
    d2 = d1[1:] - d1[:-1]
    d3 = d2[1:] - d2[:-1]
    return d3.norm(dim=-1).mean().item()


def run_episodes(policy: SmolVLAPolicy, episodes: int, steps_per_episode: int, arm_dim: int) -> dict:
    all_actions: list[torch.Tensor] = []
    combined_log: list[dict] | None = [] if policy.lac_debug_log is not None else None
    for _ in range(episodes):
        policy.reset()  # 매 에피소드마다 lac_debug_log도 새로 비워지므로, episode가 끝날 때마다 걷어간다.
        batch = build_batch(batch_size=1, arm_dim=arm_dim)
        with torch.no_grad():
            for _ in range(steps_per_episode):
                action = policy.select_action(dict(batch))
                all_actions.append(action)
        if combined_log is not None:
            combined_log.extend(policy.lac_debug_log)
    jerk = _jerk(all_actions)
    return {"jerk": jerk, "lac_debug_log": combined_log}


def summarize_debug_log(log: list[dict] | None) -> str:
    if not log:
        return "  (lac_mode=none — rho/omega/alpha/regime 없음)"
    role_aware = "alpha_stab" in log[0]
    regimes = [entry["regime"][0] for entry in log]
    regime_counts = {r: regimes.count(r) for r in ("macro", "precision", "noise")}
    rho_vals = [entry["rho"].item() for entry in log]
    omega_vals = [entry["omega"].item() for entry in log]
    lines = [
        f"  regime 비율: {regime_counts} (총 {len(regimes)}스텝)",
        f"  rho: mean={statistics.mean(rho_vals):.4f} min={min(rho_vals):.4f} max={max(rho_vals):.4f}",
        f"  omega: mean={statistics.mean(omega_vals):.4f} min={min(omega_vals):.4f} max={max(omega_vals):.4f}",
    ]
    if role_aware:
        alpha_stab = [entry["alpha_stab"].item() for entry in log]
        alpha_act = [entry["alpha_act"].item() for entry in log]
        lines.append(f"  alpha_stab: mean={statistics.mean(alpha_stab):.4f} max={max(alpha_stab):.4f}")
        lines.append(f"  alpha_act: mean={statistics.mean(alpha_act):.4f} max={max(alpha_act):.4f}")
    else:
        alpha_vals = [entry["alpha"].item() for entry in log]
        lines.append(f"  alpha: mean={statistics.mean(alpha_vals):.4f} min={min(alpha_vals):.4f} max={max(alpha_vals):.4f}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dim", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps-per-episode", type=int, default=8)
    parser.add_argument("--no-perturb-ard-heads", dest="perturb_ard_heads", action="store_false")
    parser.set_defaults(perturb_ard_heads=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()

    tiny_config = _build_tiny_smolvlm_config()
    _apply_monkeypatch(tiny_config)

    print(__doc__)
    print(
        f"설정: arm_dim={args.arm_dim} chunk_size={args.chunk_size} episodes={args.episodes} "
        f"steps_per_episode={args.steps_per_episode} perturb_ard_heads={args.perturb_ard_heads}\n"
    )

    for mode in ("none", "covla", "role_aware"):
        torch.manual_seed(0)  # 세 모드가 같은 초기 가중치/노이즈 시퀀스를 쓰도록
        policy = build_policy(args.arm_dim, args.chunk_size, mode, args.perturb_ard_heads)
        torch.manual_seed(1)
        result = run_episodes(policy, args.episodes, args.steps_per_episode, args.arm_dim)
        print(f"=== lac_mode={mode!r} ===")
        print(f"  jerk(3차 차분 크기 평균): {result['jerk']:.6f}")
        print(summarize_debug_log(result["lac_debug_log"]))
        print()

    print(
        "참고: role_aware가 covla와 숫자가 똑같이 나왔다면, 이번 섭동/시드에서는 omega가 "
        "tau_omega를 한 번도 넘지 않아서(cooperative 분기가 한 번도 안 걸려서) role_aware의 "
        "Actuator 상향 로직이 전혀 발동하지 않은 것이다 — role_aware 쪽 분기 자체가 잘못된 게 "
        "아니라(tests/test_lac.py의 role_aware 단위 테스트가 그 분기를 직접 확인한다), 이\n"
        "데모의 무작위 섭동이 우연히 그런 입력을 안 만든 것뿐이다. --no-perturb-ard-heads로도 "
        "돌려서 비교해보면(그때는 잔차가 항상 0이라 전부 macro) 두 극단을 다 볼 수 있다."
    )


if __name__ == "__main__":
    main()
