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
"""`run_closed_loop(..., per_step=...)`에 붙는 평가 로깅 유틸 (논문 Fig.4용).

정책 동작은 전혀 바꾸지 않는다 — `RolloutLogger.per_step`은 `closed_loop.py`가 액션을
`env.execute()`로 보내기 *직전*에 읽기만 할 뿐, 아무것도 되돌리거나 바꾸지 않는다(순수
사이드이펙트: 기록만).

사용 예::

    from lerobot.policies.smolvla.closed_loop import run_closed_loop
    from lerobot.policies.smolvla.eval_logging import RolloutLogger

    logger = RolloutLogger(env, method="ard_vla", episode_id="peg_in_hole_0")
    run_closed_loop(
        policy, env, decision_logic, goal=goal, start_node=start,
        per_step=logger.per_step, track_role_split=True,
    )
    logger.save("rollout_logs/ard_vla_0.npz")

물리 상태(관절 위치/속도, 엔드이펙터 위치, Stabilizer가 잡은 물체 위치, 접촉 이벤트 등)는
`env.observe()`가 반환하는 dict에서 끌어온다 — 정확한 키 이름은 환경(ARD-Gen 쪽)마다 다르므로
`physical_state_keys`로 지정한다. 지정한 키가 그 스텝의 관측에 없으면 조용히 건너뛴다(크래시
대신 — 환경마다 매 스텝 전부를 채우지 않을 수 있어서).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


DEFAULT_PHYSICAL_STATE_KEYS = (
    "joint_pos",
    "joint_vel",
    "ee_pos",
    "stabilizer_object_pos",
    "contact",
)


def _to_numpy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


class RolloutLogger:
    """`run_closed_loop`의 `per_step` 콜백 하나로 매 실행 스텝의 상태를 모아서 npz로 저장한다.

    `env`를 클로저로 들고 있다가 매 스텝 `env.observe()`를 한 번 더 호출한다(러너 자체는
    청크당 한 번만 관측한다 — 물리 상태를 매 *실행 스텝*마다 보려면 이 재호출이 필요하다).
    `env.observe()`가 매 호출마다 현재 상태를 그대로 읽어 주는(부작용 없는) 구현이라고
    가정한다 — `ClosedLoopEnv` 프로토콜 자체가 보장하는 바는 아니라서, 그렇지 않은 환경에서는
    쓰면 안 된다."""

    def __init__(
        self,
        env,
        method: str,
        episode_id: str,
        physical_state_keys: tuple[str, ...] = DEFAULT_PHYSICAL_STATE_KEYS,
    ):
        self.env = env
        self.method = method
        self.episode_id = episode_id
        self.physical_state_keys = physical_state_keys
        self.records: list[dict[str, Any]] = []

    def per_step(
        self,
        step: int,
        step_action: torch.Tensor,
        state,
        role_split_t: dict[str, torch.Tensor] | None,
    ) -> None:
        obs = dict(self.env.observe())
        record: dict[str, Any] = {
            "step": step,
            "stage": getattr(state, "current_node", None),
            "action": _to_numpy(step_action),
        }
        for key in self.physical_state_keys:
            if key in obs:
                record[key] = _to_numpy(obs[key])
        if role_split_t is not None:
            # residual_left/residual_right are exactly the Stabilizer/Actuator ARD residuals
            # under the default role map (left=Stabilizer, right=Actuator) -- see ard.py's
            # resolve_actuator_is_first/split_by_role and modeling_smolvla.py's
            # sample_actions_with_decisions_and_role_split.
            record["r_stab_norm"] = float(torch.linalg.norm(role_split_t["residual_left"]).item())
            record["r_act_norm"] = float(torch.linalg.norm(role_split_t["residual_right"]).item())
        self.records.append(record)

    def save(self, path: str) -> None:
        """레코드를 npz 하나로 저장한다. 레코드마다 키 집합이 다를 수 있어서(환경이 매 스텝
        모든 physical_state_keys를 다 채우지 않을 수 있음) 전체 레코드에서 나온 키들의 합집합을
        기준으로, 없는 스텝은 NaN(수치형)/None 자리로 채운 고정폭 배열을 만든다."""
        import os

        all_keys = set()
        for rec in self.records:
            all_keys.update(rec.keys())
        all_keys.discard("step")
        all_keys.discard("stage")

        n = len(self.records)
        out: dict[str, np.ndarray] = {
            "step": np.array([r["step"] for r in self.records], dtype=np.int64),
            "stage": np.array([str(r.get("stage")) for r in self.records]),
            "method": np.array(self.method),
            "episode_id": np.array(self.episode_id),
        }
        for key in sorted(all_keys):
            sample = next((r[key] for r in self.records if key in r), None)
            if sample is None:
                continue
            sample_arr = np.asarray(sample)
            arr = np.full((n, *sample_arr.shape), np.nan, dtype=np.float64)
            for i, rec in enumerate(self.records):
                if key in rec:
                    arr[i] = np.asarray(rec[key], dtype=np.float64)
            out[key] = arr

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, **out)

    @staticmethod
    def load(path: str) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
