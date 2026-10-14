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

환경은 duck typing 프로토콜(`ClosedLoopEnv`)로만 받는다 — 상속도 import도 필요 없고,
`observe`/`execute`/`subtask_finished`/`settle`/`safe_stop` 다섯 메서드만 있으면 된다.
`run_recovery(recovery_id: str) -> None`은 선택 메서드 — 없어도 동작하지만(아래 참고) 있으면
RECOVER 결정을 실제로 실행할 수 있다.

## 핵심 설계: "판정은 실행 *후* 관측으로, 한 박자 늦지 않게"

이전 버전은 청크를 실행하기 *전* 관측으로 Judge를 돌리고, 그 판정을 청크 실행 *후*에
`decide()`에 썼다 — 그래서 매 판정이 "지금 막 실행한 것"이 아니라 "그 이전 상태"를 채점하는
한 박자 늦은 버그가 있었다. 이번 버전은 `need_judgment` 플래그로 판정 시점을 명시적으로
관리한다:

- 서브태스크가 아직 끝나지 않았으면(`need_judgment=False`) 이번 forward의 `decisions`는
  판정에 쓰지 않고, 그 `actions`로 다음 청크를 그냥 실행한다(여러 청크짜리 서브태스크 지원 —
  청크 경계마다 실패로 오판하지 않는다).
- 서브태스크가 끝나면(`env.subtask_finished()` 또는 서브태스크 스텝 예산 초과)
  `env.settle()`을 부르고 `need_judgment=True`로 표시한다.
- 다음 iteration은 **새로** `env.observe()`(=방금 끝난 서브태스크의 결과를 반영한 관측)로
  forward를 돌리고, 그 `decisions`로 `decide()`를 부른다 — 이게 "실행 후 관측으로 판정"이다.

## prefix forward 횟수 (iteration마다 정확히 1회, 경계에 따라 누적이 다름)

매 iteration은 `predict_action_chunk_with_decisions()`를 **정확히 한 번** 부른다(로그의
`forward_calls`가 항상 1인 이유) — 그래서 "경계"를 넘는 데 몇 번의 forward가 쓰이는지는
iteration 수로 환산된다:
- 서브태스크 진행 중(청크 경계, 판정 없음): 청크당 1회.
- RETRY(같은 지시문으로 재시도): 이번 forward의 `actions`를 그대로 재사용 — 추가 forward 없음
  (경계당 1회, 판정 iteration과 재시도 실행 iteration이 같다).
- ADVANCE(다음 노드) 또는 RECOVER(복구 실행): 지시문/환경이 바뀌므로 다음 iteration에서 다시
  관측+forward해야 한다(경계당 2회 — 판정한 iteration + 다시 forward하는 iteration).

`pre_chunk`/`per_step`은 미래 기능(사전 청크 검증/상시 Stabilizer 모니터링)을 위한 자리만
만들어둔 콜백이다 — 기본 `None`이고, 실제 검증/모니터링 로직은 이번 작업 범위 밖이라 구현하지
않는다(호출부만 있다).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Protocol

import torch

from lerobot.policies.smolvla.decision_logic import DecisionKind, DecisionLogic, EpisodeState


class ClosedLoopEnv(Protocol):
    """duck typing 프로토콜 — 아래 다섯 메서드만 있으면 어떤 객체든 환경으로 쓸 수 있다.

    `run_recovery(recovery_id: str) -> None`은 선택이다(Protocol에 넣지 않음 — 구조적으로
    강제하지 않고 `hasattr`로만 확인한다): 없는 환경에서 RECOVER 결정이 나오면 경고 후 RETRY와
    동일하게 처리한다(1차 연구의 실제 복구 구현은 환경 쪽 스크립트로 나중에 채울 몫).
    """

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
    postprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    max_iterations: int = 50,
    # ponytail: task graph 노드별 필드가 아니라 러너 레벨 단일 값으로 둔다 — 서브태스크마다
    # 다른 예산이 필요해지면 그때 TaskGraphNode에 필드를 추가하면 된다(지금은 YAGNI).
    max_subtask_steps: int | None = None,
    pre_chunk: Callable[[torch.Tensor, dict, EpisodeState], None] | None = None,
    per_step: Callable[[int, torch.Tensor, EpisodeState], None] | None = None,
    post_decision: Callable[[Any, dict, EpisodeState], None] | None = None,
) -> dict[str, Any]:
    """하나의 에피소드를 끝까지(종료 노드 도달 또는 STOP) 돈다.

    반환값은 에피소드 로그 dict: `decision_history`/`judge_probabilities`(판정이 실제로 일어난
    iteration에서만 1개씩 추가), `forward_calls`/`judged`(iteration마다 1개씩 — 모듈 docstring
    참고), `retry_counts`(최종 상태), `termination_reason`.
    """
    state = EpisodeState(current_node=start_node)
    need_judgment = False
    subtask_step_count = 0
    log: dict[str, Any] = {
        "decision_history": [],
        "judge_probabilities": [],
        "forward_calls": [],
        "judged": [],
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

        log["forward_calls"].append(1)
        log["judged"].append(need_judgment)

        if need_judgment:
            decision = decision_logic.decide(decisions, state)
            history_entry = {
                "node": state.current_node,
                "kind": decision.kind.value,
                "target": decision.target,
                "reason": decision.reason,
            }
            log["decision_history"].append(history_entry)
            log["judge_probabilities"].append(_decisions_to_floats(decisions))
            if post_decision is not None:
                post_decision(decision, decisions, state)

            if decision.kind == DecisionKind.STOP:
                env.safe_stop()
                log["termination_reason"] = "stop"
                break
            if decision.kind == DecisionKind.ADVANCE and decision.target is None:
                log["termination_reason"] = "task_graph_end"
                break

            if decision.kind in (DecisionKind.RECHECK, DecisionKind.RECOVER, DecisionKind.RETRY):
                policy.cancel_pending_actions()

            if decision.kind == DecisionKind.RECHECK:
                # 행동 실행 없이 다시 관측·판정한다 — need_judgment는 True로 유지.
                env.settle()
                continue

            if decision.kind == DecisionKind.RECOVER:
                need_judgment = False
                subtask_step_count = 0
                if hasattr(env, "run_recovery"):
                    env.run_recovery(decision.target)
                    env.settle()
                    history_entry["recovery_unsupported"] = False
                    # 복구로 환경이 바뀌었으니 다시 관측+forward해서 actions를 재계산한다.
                    continue
                warnings.warn(
                    f"환경에 run_recovery()가 없어 RECOVER('{decision.target}')를 RETRY처럼 처리합니다 "
                    "— 실제 복구 구현은 환경 쪽에서 추가해야 합니다.",
                    stacklevel=2,
                )
                history_entry["recovery_unsupported"] = True
                # fallthrough — 이번 forward의 actions를 그대로 실행 (RETRY와 동일하게 추가
                # forward 없음).

            elif decision.kind == DecisionKind.ADVANCE:
                # 다음 노드로(이미 decide() 안에서 state.current_node가 갱신됨) — 지시문이
                # 바뀌므로 다시 관측+forward해서 actions를 재계산한다.
                need_judgment = False
                subtask_step_count = 0
                continue

            elif decision.kind == DecisionKind.RETRY:
                # 지시문이 같으므로 이번 forward의 actions를 그대로 실행한다 — 추가 forward 없음.
                need_judgment = False
                subtask_step_count = 0

        # ---- 실행 단계: need_judgment==False였거나, 위에서 RETRY/recovery_unsupported로
        # 넘어온 경우(둘 다 "이번 forward의 actions를 바로 실행") ----
        if pre_chunk is not None:
            pre_chunk(actions, decisions, state)
        exec_actions = postprocessor(actions) if postprocessor is not None else actions

        remaining_budget = None
        if max_subtask_steps is not None:
            remaining_budget = max(max_subtask_steps - subtask_step_count, 0)
        steps_run, finished_by_env = _execute_chunk(env, exec_actions, per_step, state, remaining_budget)
        subtask_step_count += steps_run

        budget_exceeded = (
            max_subtask_steps is not None and not finished_by_env and subtask_step_count >= max_subtask_steps
        )
        if finished_by_env or budget_exceeded:
            env.settle()
            need_judgment = True
            subtask_step_count = 0
        # else: 서브태스크가 아직 안 끝났다 — 판정도 settle()도 없이 다음 청크로 진행.
    else:
        env.safe_stop()
        log["termination_reason"] = "max_iterations_exceeded"

    log["retry_counts"] = dict(state.retry_counts)
    return log


def _execute_chunk(
    env: ClosedLoopEnv, actions: torch.Tensor, per_step, state: EpisodeState, max_steps: int | None
) -> tuple[int, bool]:
    """액션 청크(batch=1 가정, shape (1, n_steps, action_dim))를 한 스텝씩 실행하다가
    서브태스크가 끝나거나(`env.subtask_finished()`) `max_steps`(서브태스크 스텝 예산의 남은
    분량)에 닿으면 멈춘다. `(실제로 실행한 스텝 수, env.subtask_finished()로 끝났는지)`를
    돌려준다 — 후자가 False면 청크가 그냥 소진됐거나(다음 청크로 계속) 예산을 다 썼다는 뜻
    (예산 초과 판정은 호출부가 누적 스텝 수로 따로 한다)."""
    n_steps = actions.shape[1] if max_steps is None else min(actions.shape[1], max_steps)
    steps_run = 0
    for step in range(n_steps):
        step_action = actions[0, step]
        if per_step is not None:
            per_step(step, step_action, state)
        env.execute(step_action)
        steps_run += 1
        if env.subtask_finished():
            return steps_run, True
    return steps_run, False


def _decisions_to_floats(decisions: dict[str, Any]) -> dict[str, Any]:
    """`decisions`의 torch 텐서 값들을 로그에 담기 좋은 python 값(float/list)으로 변환한다."""
    out = {}
    for key, value in decisions.items():
        if hasattr(value, "tolist"):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out
