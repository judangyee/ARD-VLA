#!/usr/bin/env python3
"""ARD-VLA 2.0 폐루프 경로(`run_closed_loop` + `predict_action_chunk_with_decisions`)에서
LAC 세 모드(none/covla/role_aware)를 작은 합성 SmolVLM 백본 + 몽키패치(`tests/conftest.py`와
같은 패턴, Hugging Face Hub 접근 없음)로 실제로 몇 에피소드 돌려서 jerk, LAC 리셋 횟수/이유,
rho/omega/alpha regime 비율을 비교한다. `scripts/compare_lac_modes.py`(`select_action()`
경로)와 달리 이 스크립트는 **실제 Judge + 결정 로직 + 러너**를 끝까지 돌린다 — Judge head는
ARD 헤드처럼 zero-init이 아니라서(`judge.py` 참고) success_prob이 매 스텝 랜덤하게 나오고,
그래서 ADVANCE/RETRY/RECHECK/STOP이 실제로 섞여서 일어난다(=LAC 리셋도 episode/subtask/retry
세 이유로 실제 섞여서 찍힌다).

**중요한 주의사항** — `AsymmetricResidualHeads`(ARD 헤드)는 zero-init이라(`ard.py` 참고)
잔차가 항상 0이라서 기본적으로는 rho/omega/regime이 전부 "macro"로 결정론적이다
(`scripts/compare_lac_modes.py`와 동일한 문제). `--perturb-ard-heads`(기본 켬)로 ARD 헤드
마지막 레이어에 작은 가우시안 잡음을 주입해서(실제 학습된 가중치가 아니라 데모용 섭동)
regime이 실제로 갈리는 모습을 보여준다. Judge/결정 로직 자체는 완전히 랜덤 초기화라서
success_prob/regime 분포 둘 다 "의미 있는 수치"가 아니라 메커니즘이 끝까지 도는지 보여주는
용도다. 실제 SmolVLM2 가중치나 OpenArm 시뮬에서는 아직 전혀 돌려보지 않았다(README/docs의
"검증됨 vs 미검증" 참고)."""

from __future__ import annotations

import argparse
import logging
import statistics

import torch
import transformers
from transformers import SmolVLMConfig, SmolVLMVisionConfig

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.closed_loop import run_closed_loop
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.decision_logic import DecisionLogic
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

FAKE_MODEL_ID = "fake/tiny-smolvlm"
TASK_GRAPH = {
    "step1": {"next": "step2", "recoveries": [], "max_retries": 3},
    "step2": {"next": None, "recoveries": [], "max_retries": 3},
}


class _StubTokenizer:
    fake_image_token_id = 1
    global_image_token_id = 2


class _StubProcessor:
    tokenizer = _StubTokenizer()


def _build_tiny_smolvlm_config() -> SmolVLMConfig:
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


def build_policy(arm_dim: int, lac_mode: str, perturb_ard_heads: bool) -> SmolVLAPolicy:
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
        chunk_size=4,
        n_action_steps=4,
        max_state_dim=2 * arm_dim,
        max_action_dim=2 * arm_dim,
        use_ard=True,
        ard_arm_dim=arm_dim,
        use_judge=True,
        num_steps=2,
        lac_mode=lac_mode,
        lac_log=lac_mode != "none",
    )
    policy = SmolVLAPolicy(config)
    policy.eval()

    if perturb_ard_heads:
        gen = torch.Generator().manual_seed(0)
        for head in (policy.model.ard_heads.stabilizer_head, policy.model.ard_heads.actuator_head):
            last_linear = head[-1]
            last_linear.weight.data += 0.05 * torch.randn(last_linear.weight.shape, generator=gen)
            last_linear.bias.data += 0.05 * torch.randn(last_linear.bias.shape, generator=gen)

    return policy


def build_batch(arm_dim: int, lang_len: int = 6) -> dict:
    return {
        OBS_IMAGE: torch.rand(1, 3, 64, 64),
        OBS_STATE: torch.randn(1, 2 * arm_dim),
        OBS_LANGUAGE_TOKENS: torch.randint(0, 200, (1, lang_len)),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, lang_len, dtype=torch.bool),
    }


class FakeEnv:
    """서브태스크 하나가 execute() 3번 뒤에 끝나는 가짜 환경 — 한 "시도"당 여러 스텝이
    실행돼야 LAC가 실제로(첫 스텝은 통과, 그 다음부터 필터링) 뭔가 다른 값을 만들 여지가
    생긴다. 환경 자체의 동역학은 이 스크립트의 관심사가 아니다(LAC가 폐루프 전체를 거쳐
    실제로 동작하는지만 본다)."""

    def __init__(self, finish_after: int = 3):
        self.finish_after = finish_after
        self.execute_count_since_settle = 0
        self.executed: list[torch.Tensor] = []

    def observe(self):
        return {}

    def execute(self, action):
        self.executed.append(action.clone())
        self.execute_count_since_settle += 1

    def subtask_finished(self) -> bool:
        return self.execute_count_since_settle >= self.finish_after

    def settle(self):
        self.execute_count_since_settle = 0

    def safe_stop(self):
        pass


def _jerk(actions: list[torch.Tensor]) -> float:
    if len(actions) < 4:
        return float("nan")
    stacked = torch.stack(actions)
    d1 = stacked[1:] - stacked[:-1]
    d2 = d1[1:] - d1[:-1]
    d3 = d2[1:] - d2[:-1]
    return d3.norm(dim=-1).mean().item()


def run_episodes(policy: SmolVLAPolicy, episodes: int, arm_dim: int) -> dict:
    all_actions: list[torch.Tensor] = []
    combined_debug_log: list[dict] = []
    reset_reason_counts: dict[str, int] = {}
    termination_reasons: list[str] = []

    decision_logic = DecisionLogic(task_graph=TASK_GRAPH, tau_success=0.7, tau_fail=0.3, max_recheck=5)

    for _ in range(episodes):
        env = FakeEnv()
        with torch.no_grad():
            log = run_closed_loop(
                policy, env, decision_logic, goal="demo goal", start_node="step1",
                preprocessor=lambda obs, arm_dim=arm_dim: build_batch(arm_dim),
                max_iterations=20, log_lac_events=True,
            )
        all_actions.extend(env.executed)
        termination_reasons.append(log["termination_reason"])
        for entry in log["lac_resets"]:
            reset_reason_counts[entry["reason"]] = reset_reason_counts.get(entry["reason"], 0) + 1
        if log["lac_debug_log"]:
            combined_debug_log.extend(log["lac_debug_log"])

    return {
        "jerk": _jerk(all_actions),
        "lac_debug_log": combined_debug_log or None,
        "reset_reason_counts": reset_reason_counts,
        "termination_reasons": termination_reasons,
    }


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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no-perturb-ard-heads", dest="perturb_ard_heads", action="store_false")
    parser.set_defaults(perturb_ard_heads=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    args = parse_args()

    tiny_config = _build_tiny_smolvlm_config()
    _apply_monkeypatch(tiny_config)

    print(__doc__)
    print(f"설정: arm_dim={args.arm_dim} episodes={args.episodes} perturb_ard_heads={args.perturb_ard_heads}\n")

    for mode in ("none", "covla", "role_aware"):
        torch.manual_seed(0)
        policy = build_policy(args.arm_dim, mode, args.perturb_ard_heads)
        torch.manual_seed(1)
        result = run_episodes(policy, args.episodes, args.arm_dim)
        print(f"=== lac_mode={mode!r} ===")
        print(f"  jerk(3차 차분 크기 평균): {result['jerk']:.6f}")
        print(f"  termination_reasons: {result['termination_reasons']}")
        print(f"  lac_resets (이유별 횟수 — 'episode'는 에피소드당 1회라 episodes 수와 같음): {result['reset_reason_counts']}")
        print(summarize_debug_log(result["lac_debug_log"]))
        print()

    print(
        "참고: lac_resets에 'subtask'/'retry'가 안 보이면, 이번 랜덤 Judge 가중치로는\n"
        "success_prob이 RECHECK 구간에만 떨어져서 재관측 한도 초과(STOP)로만 끝났다는 뜻이다 —\n"
        "ADVANCE/RETRY/RECOVER 각각에서 리셋이 실제로 불리는지는 랜덤성에 의존하지 않는\n"
        "tests/test_lac_closed_loop.py의 스크립트 테스트가 직접 확인한다. role_aware가 covla와\n"
        "똑같이 나온 것도 같은 이유(omega가 tau_omega를 한 번도 못 넘음) — role_aware 분기 자체는\n"
        "tests/test_lac.py가 직접 확인했다."
    )


if __name__ == "__main__":
    main()
