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
"""ARD-VLA 폐루프 구조의 결정 로직 — task graph 상태(완료 단계/현재 노드/재시도 횟수)를 관리하고
Judge/Selector 출력을 안전 규칙으로 걸러서 다음 행동을 정하는 순수 Python 규칙 레이어.

torch 의존성이 전혀 없다 — `decide()`에 넘기는 `decisions` dict의 값들(`success_prob` 등)이
torch 텐서여도 `.item()`/`.argmax()` 같은 duck-typing으로만 다루고, 이 파일 자체는 `import
torch`를 하지 않는다(그래서 GPU/Hub 환경 없이도, 심지어 torch가 설치되지 않은 환경에서도 이
모듈만 독립적으로 쓸 수 있다).

안전 규칙(허용되지 않은 전이 거부, 재시도/재관측 한도 초과 시 STOP)은 Judge/Selector가 꺼져
있거나(`use_selector=False`) 이상한 값을 내더라도 항상 적용된다 — Selector는 "이 결정 로직이
이미 허용한 후보 중 어느 것을 고를지"에만 영향을 준다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DecisionKind(Enum):
    ADVANCE = "advance"  # 다음 노드로( node.next ). next가 None이면 task graph 종료.
    RETRY = "retry"  # 같은 노드를 그대로 다시 시도 (실패 유형을 모를 때의 rule table 기본값)
    RECOVER = "recover"  # node.recoveries 중 하나의 복구 동작을 실행
    RECHECK = "recheck"  # 재관측(re-observe) — 성공/실패 둘 다 아직 확신할 수 없는 중간 구간
    STOP = "stop"  # 안전 정지 — 재시도/재관측 한도 초과


@dataclass
class Decision:
    kind: DecisionKind
    target: str | None = None  # ADVANCE -> 다음 노드 id(또는 종료면 None), RECOVER -> 복구 동작 id
    reason: str = ""


@dataclass
class TaskGraphNode:
    id: str
    next: str | None = None
    recoveries: list[str] = field(default_factory=list)
    max_retries: int = 3


@dataclass
class EpisodeState:
    """하나의 에피소드(폐루프 러너 1회 실행) 동안의 task graph 진행 상태."""

    current_node: str
    completed: list[str] = field(default_factory=list)
    retry_counts: dict[str, int] = field(default_factory=dict)
    recheck_counts: dict[str, int] = field(default_factory=dict)


def load_task_graph(graph: dict) -> dict[str, TaskGraphNode]:
    """dict(또는 YAML에서 읽은 dict)를 `{node_id: TaskGraphNode}`로 변환한다.

    스키마: `{node_id: {"next": str|None, "recoveries": [str, ...], "max_retries": int}}`
    (YAML 파일을 쓰려면 호출부에서 `yaml.safe_load(path.read_text())`로 미리 dict를 만들어
    넘기면 된다 — 이 모듈은 YAML 파서에 의존하지 않는다).
    """
    nodes = {}
    for node_id, spec in graph.items():
        nodes[node_id] = TaskGraphNode(
            id=node_id,
            next=spec.get("next"),
            recoveries=list(spec.get("recoveries", [])),
            max_retries=spec.get("max_retries", 3),
        )
    return nodes


def _argmax(values) -> int | None:
    """`values`가 1차원이든(batch 없음) 2차원(batch, N)이든, torch 텐서든 plain list/tuple든
    상관없이 그 중 확률이 가장 큰 인덱스를 돌려준다. torch를 import하지 않고 duck-typing만
    쓴다 — `polling batch_size=1` 가정(이 모듈을 쓰는 closed_loop.py 러너는 환경 하나를 한 번에
    하나씩 돈다)."""
    if values is None:
        return None
    if hasattr(values, "dim"):  # torch.Tensor
        if values.dim() > 1:
            values = values[0]
        return int(values.argmax().item())
    if len(values) > 0 and hasattr(values[0], "__len__"):
        values = values[0]  # batch 차원이 있는 list-of-list
    return max(range(len(values)), key=lambda i: values[i])


def _scalar(value) -> float | None:
    """`value`가 torch 텐서(스칼라 또는 size-1 배치)든 plain float든 python float로 변환한다."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class DecisionLogic:
    """task graph + 안전 규칙 + (선택적) Judge/Selector 출력으로 다음 행동을 결정한다.

    `use_selector=False`(Selector 끔)일 때도 이 클래스 하나로 폐루프가 동작해야 한다는 게
    설계 요구사항이다 — Selector가 없으면 `decide()`가 그냥 `failure_type_to_recovery` rule
    table만으로 복구 동작을 고른다(Selector의 "기본값"으로도 동작).
    """

    def __init__(
        self,
        task_graph: dict,
        option_vocab: list[str] | None = None,
        # tau_success/tau_fail 기본값은 placeholder다 — 실제 분리 지점은 실험(검증 데이터셋에서
        # Judge 캘리브레이션 곡선을 보고)으로 정해야 한다. 이번 작업(구조 구현)은 그 실측을
        # 포함하지 않는다.
        tau_success: float = 0.7,
        tau_fail: float = 0.3,
        max_recheck: int = 3,
        failure_type_to_recovery: dict[int, str] | None = None,
    ):
        if tau_fail > tau_success:
            raise ValueError(f"tau_fail({tau_fail})이 tau_success({tau_success})보다 클 수 없습니다.")
        self.nodes = load_task_graph(task_graph)
        self.option_vocab = option_vocab or []
        self.tau_success = tau_success
        self.tau_fail = tau_fail
        self.max_recheck = max_recheck
        self.failure_type_to_recovery = failure_type_to_recovery or {}

    def decide(self, decisions: dict, state: EpisodeState) -> Decision:
        """`decisions`는 `predict_action_chunk_with_decisions()`가 반환하는 dict(또는 같은
        키 구조를 가진 어떤 것) — `success_prob`가 반드시 있어야 한다(Judge가 꺼져 있으면
        이 메서드를 호출할 수 없다, use_judge=False일 때는 폐루프 자체를 쓸 수 없다는 뜻).
        """
        node = self.nodes[state.current_node]
        p = _scalar(decisions.get("success_prob"))
        if p is None:
            raise ValueError(
                "decisions에 success_prob가 없습니다 — use_judge=True인 정책으로 predict_action_chunk_with_decisions를 호출해야 합니다."
            )

        if p >= self.tau_success:
            return self._advance(node, state)
        if p < self.tau_fail:
            return self._handle_failure(decisions, node, state)
        return self._recheck(node, state)

    def _advance(self, node: TaskGraphNode, state: EpisodeState) -> Decision:
        state.completed.append(node.id)
        state.retry_counts.pop(node.id, None)
        state.recheck_counts.pop(node.id, None)
        if node.next is None:
            return Decision(DecisionKind.ADVANCE, target=None, reason="task graph 종료 노드에 도달")
        state.current_node = node.next
        return Decision(DecisionKind.ADVANCE, target=node.next, reason="success_prob >= tau_success")

    def _handle_failure(self, decisions: dict, node: TaskGraphNode, state: EpisodeState) -> Decision:
        retry_count = state.retry_counts.get(node.id, 0)
        if retry_count >= node.max_retries:
            return Decision(DecisionKind.STOP, reason=f"'{node.id}' 재시도 한도({node.max_retries}) 초과")

        state.retry_counts[node.id] = retry_count + 1
        recovery = self._choose_recovery(decisions, node)
        if recovery is None:
            return Decision(DecisionKind.RETRY, target=node.id, reason="실패 유형 미확인 — 같은 노드 재시도(rule table 기본값)")
        return Decision(DecisionKind.RECOVER, target=recovery, reason="rule table/Selector가 고른 복구 동작")

    def _choose_recovery(self, decisions: dict, node: TaskGraphNode) -> str | None:
        """Selector 출력이 있으면 그걸 우선 쓰되, 반드시 `node.recoveries`(안전 규칙 — 허용된
        전이 목록) 안에 있는지 확인한다. 벗어나면 Selector를 무시하고 rule table로 대체한다
        — Judge/Selector가 뭘 내놓든 이 검증은 항상 적용된다."""
        selector_probs = decisions.get("selector_probs")
        if selector_probs is not None and self.option_vocab:
            idx = _argmax(selector_probs)
            if idx is not None and 0 <= idx < len(self.option_vocab):
                candidate = self.option_vocab[idx]
                if candidate in node.recoveries:
                    return candidate
        return self._rule_table_recovery(decisions, node)

    def _rule_table_recovery(self, decisions: dict, node: TaskGraphNode) -> str | None:
        """실패 유형(`failure_probs`의 argmax) -> 복구 동작 매핑. 실패 유형이 없거나(Judge가
        failure head를 안 쓰는 경우), 매핑된 복구 동작이 이 노드에서 허용되지 않으면 None을
        돌려줘서 `_handle_failure`가 RETRY로 떨어지게 한다."""
        failure_probs = decisions.get("failure_probs")
        failure_type = _argmax(failure_probs)
        if failure_type is None:
            return None
        recovery = self.failure_type_to_recovery.get(failure_type)
        if recovery is not None and recovery in node.recoveries:
            return recovery
        return None

    def _recheck(self, node: TaskGraphNode, state: EpisodeState) -> Decision:
        count = state.recheck_counts.get(node.id, 0)
        if count >= self.max_recheck:
            return Decision(DecisionKind.STOP, reason=f"'{node.id}' 재관측 한도({self.max_recheck}) 초과")
        state.recheck_counts[node.id] = count + 1
        return Decision(DecisionKind.RECHECK, target=node.id, reason="tau_fail <= success_prob < tau_success")

    def allowed_mask(self, state: EpisodeState) -> list[bool]:
        """현재 노드에서 허용되는 옵션(= `node.recoveries`에 속하는 것)만 True — stage 2의
        `predict_action_chunk_with_decisions(selector_mask=...)`에 넘길 마스크를 만든다
        (호출부가 `torch.tensor(mask, dtype=torch.bool)`로 감싸야 한다 — 이 모듈은 torch를
        모른다)."""
        node = self.nodes[state.current_node]
        allowed = set(node.recoveries)
        return [option in allowed for option in self.option_vocab]

    def instruction_text(self, goal: str, state: EpisodeState) -> str:
        """"목표 + 현재 서브태스크 + 완료한 단계"를 하나의 지시문 문자열로 합친다 — 폐루프
        러너가 이 문자열을 `complementary_data["task"]`에 넣어 기존 전처리 파이프라인
        (`processor_smolvla.py`의 `make_smolvla_pre_post_processors`)에 그대로 통과시킨다
        (PROGRESS.md 폐루프 1단계 추적 기록의 항목 4 참고)."""
        node = self.nodes[state.current_node]
        completed_str = ", ".join(state.completed) if state.completed else "none"
        return f"{goal} | current subtask: {node.id} | completed: {completed_str}"
