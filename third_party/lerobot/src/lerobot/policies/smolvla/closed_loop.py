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

## LAC(ARD-VLA 2.0) — 실제 실행 스텝마다, 리셋 규칙과 함께

`policy.config.lac_mode != "none"`이면 `predict_action_chunk_with_decisions()`가
`decisions["_lac_role_split"]`로 공유/잔차 분해를 함께 돌려준다(Judge/Selector 판정과
무관한 내부 배관이라 `decide()`에 넘기기 전에 꺼내 쓴다). LAC는 **chunk 단위가 아니라
실제 실행 스텝마다** 적용한다 — `_execute_chunk()`가 `env.execute()` 직전에
`policy.apply_lac_step()`을 거친다(`select_action()`의 청크 단위 사전 필터링과 같은
`lac.py` 코드를 공유하되, 적용 시점만 다르다).

LAC는 "이전 명령을 기억하고 천천히 따라가는 필터"라서, 리셋 없이 넘어가면 새 서브태스크나
복구 동작이 이전 궤적 쪽으로 끌려간다. 그래서 `policy._lac.reset()`을 다음 세 시점에 부른다
(리셋 직후 첫 스텝은 필터링 없이 그대로 통과 — `lac.py`의 `q̃_0 = q_1` 규칙):
- **에피소드 시작**: `run_closed_loop()`가 맨 처음에 `policy.reset()`(큐까지 포함한 전체
  리셋)을 부른다.
- **Selector가 다음 서브태스크로 전환할 때**(`ADVANCE`, target이 다음 노드): `policy._lac`만
  리셋(큐/디버그 로그는 그대로 — 에피소드 전체에 걸쳐 누적).
- **Judge가 실패 판정 → 재시도/복구로 들어갈 때**(`RETRY`/`RECOVER`, `run_recovery`가 없어서
  RETRY로 대체되는 경우 포함): 역시 `policy._lac`만 리셋.

`RECHECK`(아직 불확실 — 같은 시도를 다시 관측만 하는 것)은 리셋하지 않는다 — 진행 중인
시도를 계속 지켜보는 것이라 필터 상태를 끊을 이유가 없다.

`log_lac_events=True`면 반환 로그에 `lac_resets`(각 리셋의 iteration/이유:
"episode"/"subtask"/"retry")와 `lac_debug_log`(`policy.lac_debug_log`— 매 실행 스텝의
rho/omega/alpha/regime)를 채운다.
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


def _reset_lac(policy, log: dict[str, Any], iteration: int, reason: str, log_lac_events: bool) -> None:
    """`policy._lac`(있으면, 즉 `lac_mode != "none"`)만 리셋한다 — `policy.reset()`(큐/디버그
    로그까지 전부 초기화하는 전체 리셋)과 달리 에피소드 중간의 서브태스크 전환/재시도·복구
    진입용이다. 둘 다 없으면(`policy._lac is None`) 조용히 아무 것도 안 한다."""
    if getattr(policy, "_lac", None) is not None:
        policy._lac.reset()
    if log_lac_events:
        log["lac_resets"].append({"iteration": iteration, "reason": reason})


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
    log_lac_events: bool = False,
) -> dict[str, Any]:
    """하나의 에피소드를 끝까지(종료 노드 도달 또는 STOP) 돈다. 맨 처음 `policy.reset()`을
    불러서 큐/LAC 상태를 포함해 완전히 새 에피소드로 시작한다(모듈 docstring의 "LAC" 절 참고).

    반환값은 에피소드 로그 dict: `decision_history`/`judge_probabilities`(판정이 실제로 일어난
    iteration에서만 1개씩 추가), `forward_calls`/`judged`(iteration마다 1개씩 — 모듈 docstring
    참고), `retry_counts`(최종 상태), `termination_reason`, `lac_resets`/`lac_debug_log`
    (`log_lac_events=True`일 때만 채워짐).
    """
    policy.reset()
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
        "lac_resets": [],
        "lac_debug_log": None,
    }
    _reset_lac(policy, log, iteration=-1, reason="episode", log_lac_events=log_lac_events)

    for iteration in range(max_iterations):
        instruction = decision_logic.instruction_text(goal, state)
        observation = dict(env.observe())
        observation["task"] = instruction
        batch = preprocessor(observation) if preprocessor is not None else observation

        selector_mask = _selector_mask_tensor(decision_logic.allowed_mask(state))
        with torch.no_grad():
            actions, decisions = policy.predict_action_chunk_with_decisions(batch, selector_mask=selector_mask)
        # Judge/Selector의 진짜 판정이 아닌 LAC 내부 배관이라, decide()/로깅이 보기 전에 꺼낸다
        # (요청 (4): "Judge/Selector의 입력과 판정 로직은 건드리지 마라").
        role_split = decisions.pop("_lac_role_split", None)

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
                # 행동 실행 없이 다시 관측·판정한다 — need_judgment는 True로 유지. 아직 불확실한
                # 같은 시도를 계속 지켜보는 것이라 LAC는 리셋하지 않는다.
                env.settle()
                continue

            if decision.kind == DecisionKind.RECOVER:
                need_judgment = False
                subtask_step_count = 0
                _reset_lac(policy, log, iteration, "retry", log_lac_events)
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
                # 바뀌므로 다시 관측+forward해서 actions를 재계산한다. 새 서브태스크로 넘어가는
                # 것이므로 LAC가 이전 서브태스크의 궤적을 끌고 가지 않도록 리셋한다.
                need_judgment = False
                subtask_step_count = 0
                _reset_lac(policy, log, iteration, "subtask", log_lac_events)
                continue

            elif decision.kind == DecisionKind.RETRY:
                # 지시문이 같으므로 이번 forward의 actions를 그대로 실행한다 — 추가 forward 없음.
                # 같은 노드를 처음부터 다시 시도하는 것이므로 LAC도 리셋한다.
                need_judgment = False
                subtask_step_count = 0
                _reset_lac(policy, log, iteration, "retry", log_lac_events)

        # ---- 실행 단계: need_judgment==False였거나, 위에서 RETRY/recovery_unsupported로
        # 넘어온 경우(둘 다 "이번 forward의 actions를 바로 실행") ----
        if pre_chunk is not None:
            pre_chunk(actions, decisions, state)
        exec_actions = postprocessor(actions) if postprocessor is not None else actions

        remaining_budget = None
        if max_subtask_steps is not None:
            remaining_budget = max(max_subtask_steps - subtask_step_count, 0)
        steps_run, finished_by_env = _execute_chunk(
            policy, env, exec_actions, role_split, per_step, state, remaining_budget
        )
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
    if log_lac_events:
        log["lac_debug_log"] = policy.lac_debug_log
    return log


def _execute_chunk(
    policy,
    env: ClosedLoopEnv,
    actions: torch.Tensor,
    role_split: dict[str, torch.Tensor] | None,
    per_step,
    state: EpisodeState,
    max_steps: int | None,
) -> tuple[int, bool]:
    """액션 청크(batch=1 가정, shape (1, n_steps, action_dim))를 한 스텝씩 실행하다가
    서브태스크가 끝나거나(`env.subtask_finished()`) `max_steps`(서브태스크 스텝 예산의 남은
    분량)에 닿으면 멈춘다. `(실제로 실행한 스텝 수, env.subtask_finished()로 끝났는지)`를
    돌려준다 — 후자가 False면 청크가 그냥 소진됐거나(다음 청크로 계속) 예산을 다 썼다는 뜻
    (예산 초과 판정은 호출부가 누적 스텝 수로 따로 한다).

    `role_split`이 있으면(`lac_mode != "none"`) `env.execute()` 바로 직전에 그 스텝의 공유/
    잔차 성분으로 `policy.apply_lac_step()`을 거친다 — chunk 단위가 아니라 실제 실행 스텝마다
    LAC를 적용하라는 요구사항이라, 여기서 액션을 꺼내 보내기 직전에만 필터링한다."""
    n_steps = actions.shape[1] if max_steps is None else min(actions.shape[1], max_steps)
    steps_run = 0
    for step in range(n_steps):
        step_action = actions[:, step]
        if role_split is not None:
            role_split_t = {key: value[:, step] for key, value in role_split.items()}
            step_action = policy.apply_lac_step(step_action, role_split_t)
        step_action = step_action[0]
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
