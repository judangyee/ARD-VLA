# Copyright 2026 The ARD-VLA authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ARD-VLA 폐루프 구조의 최소 환경-무관(environment-agnostic) 러너.

환경은 duck typing 프로토콜(`ClosedLoopEnv`) 하나로만 받는다 — 상속도 import도 필요 없고,
`observe`/`execute`/`subtask_finished`/`settle`/`safe_stop` 다섯 메서드만 있으면 된다.

루프 한 iteration:
  1. `decision_logic.instruction_text(goal, state)`로 지시문 생성(목표+현재 서브태스크+완료
     단계, 1단계 추적 기록의 항목 4 참고).
  2. `env.observe()` + 지시문을 `preprocessor`(기존 `make_smolvla_pre_post_processors`
     전처리 파이프라인과 같은 역할 — 토큰화 등)로 정책이 바로 쓸 수 있는 배치로 변환.
  3. `predict_action_chunk_with_decisions()`를 **한 번** 호출해서 prefix forward 하나로
     "지금 실행할 액션 청크"와 "지금 보고 있는 상태에 대한 Judge/Selector 판정"을 동시에
     얻는다 — 이번 iteration의 `decisions`는 (두 번째 iteration부터는) 바로 이전 iteration이
     실행+재관측한 결과를 판정한 것이다.
  4. 그 액션 청크를 `env.execute()`로 한 스텝씩 실행하다가 `env.subtask_finished()`가 True가
     되거나 청크가 끝나면 멈춘다.
  5. `env.settle()`(정지/안정화) 후 다음 iteration을 위해 다시 `env.observe()`할 준비를 한다.
  6. 3에서 이미 얻은 `decisions`로 `decision_logic.decide()`를 불러 ADVANCE/RETRY/RECOVER/
     RECHECK/STOP을 받는다. ADVANCE가 아니면(실패/재관측) `policy.cancel_pending_actions()`로
     남은 액션을 버린다 — ponytail: 이 구현은 액션 청크를 큐에 쌓지 않고 바로 한 번에 실행하기
     때문에 이 호출이 "버릴" 실제 큐 항목은 없지만, select_action()류의 큐 기반 실행으로
     바꾸거나 다른 호출부가 섞여도 안전하도록(그리고 요구사항을 문자 그대로 지키도록) 항상
     호출한다. 큐 기반으로 바꾸고 싶으면(실시간 스트리밍 실행) 이 지점에 `_queues[ACTION]`을
     직접 다루는 코드로 교체하면 된다.
  task graph의 종료 노드(ADVANCE, target=None)에 도달하거나 STOP을 받으면 끝난다.

`pre_chunk`/`per_step`/`post_decision`은 미래 기능(사전 청크 검증/상시 Stabilizer 모니터링/
로깅-복구데이터 수집)을 위한 자리만 만들어둔 콜백이다 — 기본 `None`이고, 실제 검증/모니터링
로직은 이번 작업 범위 밖이라 구현하지 않는다(호출부만 있다).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

import torch

from lerobot.policies.smolvla.decision_logic import DecisionKind, DecisionLogic, EpisodeState


class ClosedLoopEnv(Protocol):
    """duck typing 프로토콜 — 이 다섯 메서드만 있으면 어떤 객체든 환경으로 쓸 수 있다."""

    def observe(self) -> dict[str, Any]: ...
    def execute(self, action: Any) -> None: ...
    def subtask_finished(self) -> bool: ...
    def settle(self) -> None: ...
    def safe_stop(self) -> None: ...


def _selector_mask_tensor(mask: list[bool] | None) -> torch.Tensor | None:
    if not mask:
        return None
    return torch.tensor(mask, dtype=torch.bool)


def run_closed_loop(
    policy,
    env: ClosedLoopEnv,
    decision_logic: DecisionLogic,
    goal: str,
    start_node: str,
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    max_iterations: int = 50,
    pre_chunk: Callable[[torch.Tensor, dict, EpisodeState], None] | None = None,
    per_step: Callable[[int, torch.Tensor, EpisodeState], None] | None = None,
    post_decision: Callable[[Any, dict, EpisodeState], None] | None = None,
) -> dict[str, Any]:
    """하나의 에피소드를 끝까지(종료 노드 도달 또는 STOP) 돈다. 반환값은 에피소드 로그 dict —
    `decision_history`(각 iteration의 Decision), `judge_probabilities`(각 iteration의
    `decisions`를 python 스칼라로 정리한 것), `retry_counts`(최종 상태), `termination_reason`.
    """
    state = EpisodeState(current_node=start_node)
    log: dict[str, Any] = {
        "decision_history": [],
        "judge_probabilities": [],
        "retry_counts": {},
        "termination_reason": None,
    }

    for _ in range(max_iterations):
        instruction = decision_logic.instruction_text(goal, state)
        observation = dict(env.observe())
        observation["task"] = instruction
        batch = preprocessor(observation) if preprocessor is not None else observation

        selector_mask = _selector_mask_tensor(decision_logic.allowed_mask(state))
        with torch.no_grad():
            actions, decisions = policy.predict_action_chunk_with_decisions(batch, selector_mask=selector_mask)

        if pre_chunk is not None:
            pre_chunk(actions, decisions, state)

        _execute_chunk(env, actions, per_step, state)
        env.settle()

        decision = decision_logic.decide(decisions, state)
        if post_decision is not None:
            post_decision(decision, decisions, state)

        log["decision_history"].append(
            {"node": state.current_node, "kind": decision.kind.value, "target": decision.target, "reason": decision.reason}
        )
        log["judge_probabilities"].append(_decisions_to_floats(decisions))

        if decision.kind != DecisionKind.ADVANCE:
            # 실패/재관측 — 이번에 받은 액션 청크(이미 다 실행했거나 일부만 실행한) 중 남은
            # 부분을 버린다(모듈 docstring의 ponytail 참고).
            policy.cancel_pending_actions()

        if decision.kind == DecisionKind.STOP:
            env.safe_stop()
            log["termination_reason"] = "stop"
            break
        if decision.kind == DecisionKind.ADVANCE and decision.target is None:
            log["termination_reason"] = "task_graph_end"
            break
    else:
        env.safe_stop()
        log["termination_reason"] = "max_iterations_exceeded"

    log["retry_counts"] = dict(state.retry_counts)
    return log


def _execute_chunk(env: ClosedLoopEnv, actions: torch.Tensor, per_step, state: EpisodeState) -> None:
    """액션 청크(batch=1 가정, shape (1, n_steps, action_dim))를 한 스텝씩 실행하다가
    서브태스크가 끝나면 멈춘다."""
    n_steps = actions.shape[1]
    for step in range(n_steps):
        step_action = actions[0, step]
        if per_step is not None:
            per_step(step, step_action, state)
        env.execute(step_action)
        if env.subtask_finished():
            break


def _decisions_to_floats(decisions: dict[str, Any]) -> dict[str, Any]:
    """`decisions`의 torch 텐서 값들을 로그에 담기 좋은 python 값(float/list)으로 변환한다."""
    out = {}
    for key, value in decisions.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out
