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
"""Co-VLA(arXiv 2606.20285)의 Latent-Aware Controller(LAC) — 학습이 필요 없는 추론 시점
EMA 필터. ARD의 "공유(action_out_proj 출력) + 잔차(ARD 헤드 출력)" 가산 구조(A안 — PROGRESS.md
의 LAC 0단계 확인 기록 참고)에서, 팔별 공유/잔차 성분의 에너지 비율(ρ)과 협동 방향(ω)을 보고
매 실행 스텝마다 EMA 필터 강도 α를 조절해서 액션 떨림을 줄인다.

두 모드:
- `role_aware=False`(Co-VLA 원본, "covla"): 두 팔에 같은 α를 쓴다.
- `role_aware=True`("role_aware", ARD 확장): ARD의 Stabilizer/Actuator 역할별로 α를 따로
  둔다 — Stabilizer는 α 상한을 낮게 고정해 항상 강하게 평활화하고, Actuator는 반대 방향
  협동(ω가 큼, 예: 뚜껑 비틀기처럼 두 팔이 서로 다른 방향으로 힘을 주는 상황)일 때만 α를
  높여 빠르게 통과시킨다.

그리퍼 채널(각 팔의 마지막 `gripper_dim`개)은 ρ/ω 계산과 필터링 양쪽에서 항상 제외한다 —
pass-through(입력을 그대로 출력, 필터를 전혀 거치지 않음).

`mirror_sign`: OpenArm 좌우 팔은 일부 관절의 축 방향(또는 부모 프레임)이 미러링돼 있어서,
같은 물리적 포즈라도 왼팔/오른팔 관절각 부호가 반대로 나올 수 있다 — 보정 없이 ω(협동
cosine)를 계산하면 부호가 뒤집힌다. 기본값 `[-1, -1, 1, 1, 1, -1, 1]`(joint1~7)은
ARD-Gen의 `assets/openarm/openarm_bimanual.xml`에서 좌우 joint의 `axis`/`range` 속성을
직접 비교해서 얻은 것이다 — **MJCF 텍스트 비교일 뿐, 실제 forward kinematics로 "왼팔 q,
오른팔 -q가 진짜 거울상 포즈"인지 시뮬레이션에서 확인하지는 않았다(미검증, README의
"검증됨 vs 미검증" 참고)**. 이 보정은 물리적 성질이라 항상 "오른팔" 쪽에 적용되고, 역할
(Stabilizer/Actuator) 배정과는 무관하다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LACConfig:
    # --- Co-VLA 원본 기본값 ---
    alpha_base: float = 0.4
    delta_macro: float = 0.4
    delta_prec: float = 0.3
    delta_noise: float = 0.1
    tau_rho: float = 0.01
    tau_omega: float = 0.3
    beta: float = 0.2
    alpha_min: float = 0.05
    alpha_max: float = 0.95
    epsilon: float = 1e-6
    # --- 역할 인식(role_aware) 확장 전용 ---
    role_aware: bool = False
    alpha_stab_max: float = 0.5
    alpha_act_cooperative: float = 0.9
    # 관절 개수만큼(그리퍼 제외) — None이면 전부 1.0(미러링 없음으로 취급).
    mirror_sign: list[float] | None = None


class LatentAwareController:
    """상태(q̃_{t-1}, 역할별 α_{t-1})를 들고 있는 팔별 EMA 필터. `reset()`으로 에피소드마다
    초기화한다. `step()`은 실행 스텝 하나(= chunk에서 꺼낸 액션 하나)에 대응한다."""

    def __init__(
        self,
        config: LACConfig,
        joint_dim: int,
        gripper_dim: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.joint_dim = joint_dim
        self.gripper_dim = gripper_dim
        self.dtype = dtype
        mirror = config.mirror_sign if config.mirror_sign is not None else [1.0] * joint_dim
        if len(mirror) != joint_dim:
            raise ValueError(f"mirror_sign 길이({len(mirror)})가 joint_dim({joint_dim})과 다릅니다.")
        self._mirror_sign = torch.tensor(mirror, dtype=dtype)
        self.reset()

    def reset(self) -> None:
        """에피소드(또는 새 chunk 시퀀스) 시작 시 호출 — q̃_{t-1}/α_{t-1} 상태를 전부 지운다.
        다음 `step()` 호출이 "첫 스텝"으로 취급되어 `q̃_0 = q_1`이 된다."""
        self._q_prev_j: Tensor | None = None  # (batch, 2*joint_dim), [left_joints, right_joints]
        self._alpha_prev = {"stab": None, "act": None} if self.config.role_aware else None

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """마지막 축을 (joint, gripper)로 나눈다. `gripper_dim=0`이면 gripper는 빈 텐서."""
        if self.gripper_dim == 0:
            return x, x[..., :0]
        return x[..., : -self.gripper_dim], x[..., -self.gripper_dim :]

    @staticmethod
    def _cosine_coop(a: Tensor, b: Tensor, eps: float) -> Tensor:
        """ω = -<a,b> / (‖a‖‖b‖ + ε) — 반대 방향(cos=-1)일 때 ω가 커진다(협동/정밀 신호)."""
        dot = (a * b).sum(dim=-1)
        return -dot / (a.norm(dim=-1) * b.norm(dim=-1) + eps)

    def _regime(self, rho: Tensor, omega: Tensor) -> tuple[Tensor, list[str]]:
        cfg = self.config
        macro = rho < cfg.tau_rho
        precision = (~macro) & (omega > cfg.tau_omega)
        alpha_hat = torch.where(
            macro,
            torch.full_like(rho, cfg.alpha_base + cfg.delta_macro),
            torch.where(
                precision,
                torch.full_like(rho, cfg.alpha_base + cfg.delta_prec),
                torch.full_like(rho, cfg.alpha_base - cfg.delta_noise),
            ),
        )
        regime = [
            "macro" if m else ("precision" if p else "noise")
            for m, p in zip(macro.tolist(), precision.tolist(), strict=True)
        ]
        return alpha_hat, regime

    @staticmethod
    def _resolve_role(x_left: Tensor, x_right: Tensor, stabilizer_is_left: bool | Tensor) -> tuple[Tensor, Tensor]:
        """(left, right) -> (stabilizer, actuator)."""
        if isinstance(stabilizer_is_left, bool):
            return (x_left, x_right) if stabilizer_is_left else (x_right, x_left)
        mask = stabilizer_is_left.view(-1, *([1] * (x_left.ndim - 1))).to(dtype=torch.bool)
        return torch.where(mask, x_left, x_right), torch.where(mask, x_right, x_left)

    @staticmethod
    def _unresolve_role(x_stab: Tensor, x_act: Tensor, stabilizer_is_left: bool | Tensor) -> tuple[Tensor, Tensor]:
        """(stabilizer, actuator) -> (left, right) — `_resolve_role`의 역연산."""
        if isinstance(stabilizer_is_left, bool):
            return (x_stab, x_act) if stabilizer_is_left else (x_act, x_stab)
        mask = stabilizer_is_left.view(-1, *([1] * (x_stab.ndim - 1))).to(dtype=torch.bool)
        return torch.where(mask, x_stab, x_act), torch.where(mask, x_act, x_stab)

    def _role_alpha_hat(self, alpha_hat_shared: Tensor, omega_role: Tensor) -> tuple[Tensor, Tensor]:
        """공유 regime 추정치(α̂_shared)로부터 역할별 상한/협동 override를 적용한다.
        Stabilizer: 상한만 `alpha_stab_max`로 낮춤(항상 강한 평활화). Actuator: 협동
        (ω_role > τ_ω)일 때만 `alpha_act_cooperative`로 덮어쓰고, 아니면(macro/noise 둘 다)
        공유 추정치를 그대로 쓴다 — "떨림 판정일 때만 낮춤"이 자동으로 성립한다(noise 분기는
        이미 alpha_hat_shared 자체가 낮다)."""
        cfg = self.config
        alpha_hat_stab = alpha_hat_shared.clamp(cfg.alpha_min, cfg.alpha_stab_max)
        is_cooperative = omega_role > cfg.tau_omega
        alpha_hat_act_raw = torch.where(
            is_cooperative, torch.full_like(alpha_hat_shared, cfg.alpha_act_cooperative), alpha_hat_shared
        )
        alpha_hat_act = alpha_hat_act_raw.clamp(cfg.alpha_min, cfg.alpha_max)
        return alpha_hat_stab, alpha_hat_act

    def step(
        self,
        a_shared_L: Tensor,
        a_shared_R: Tensor,
        a_res_L: Tensor,
        a_res_R: Tensor,
        q_t: Tensor,
        stabilizer_is_left: bool | Tensor = True,
    ) -> tuple[Tensor, dict]:
        """`a_shared_{L,R}`/`a_res_{L,R}`: (batch, arm_dim) — 그 팔의 공유/잔차 성분(그리퍼
        채널 포함, 내부에서 제외하고 계산). `q_t`: (batch, 2*arm_dim) — 실제로 필터링할 값
        (left 블록 다음 right 블록). `stabilizer_is_left`: bool 또는 (batch,) bool 텐서 —
        기본 True(왼팔=Stabilizer, `ard_default_actuator_arm="right"`와 동일한 전제). role
        classifier 출력이 있으면 그 결과를 여기 넣으면 된다(이 모듈 자체는 classifier를
        구현하지 않는다 — 현재 ARD-VLA에 그런 classifier가 없다, PROGRESS.md 참고).

        반환: `(q̃_t, debug)` — `debug`는 `rho`/`omega`/`regime`(배치별 리스트)과, covla면
        `alpha_hat`/`alpha`, role_aware면 `alpha_hat_stab`/`alpha_hat_act`/`alpha_stab`/
        `alpha_act`를 담은 dict.
        """
        cfg = self.config
        eps = cfg.epsilon
        arm_dim = self.joint_dim + self.gripper_dim

        shared_L_j, _ = self._split(a_shared_L)
        shared_R_j, _ = self._split(a_shared_R)
        res_L_j, _ = self._split(a_res_L)
        res_R_j, _ = self._split(a_res_R)
        mirror = self._mirror_sign.to(device=res_R_j.device, dtype=res_R_j.dtype)
        res_R_j = res_R_j * mirror  # 물리적 보정 — 역할과 무관하게 항상 오른팔에 적용

        E_s = 0.5 * (shared_L_j.norm(dim=-1) + shared_R_j.norm(dim=-1))
        E_r = 0.5 * (res_L_j.norm(dim=-1) + res_R_j.norm(dim=-1))
        rho = E_r / (E_s + eps)
        omega_lr = self._cosine_coop(res_L_j, res_R_j, eps)
        alpha_hat_shared, regime = self._regime(rho, omega_lr)

        q_L, q_R = q_t[..., :arm_dim], q_t[..., arm_dim : 2 * arm_dim]
        q_L_j, q_L_g = self._split(q_L)
        q_R_j, q_R_g = self._split(q_R)

        first_step = self._q_prev_j is None
        if first_step:
            # q̃_0 = q_1 — 부동소수점에서는 (1-a)*q+a*q가 q와 비트 단위로 같지 않을 수 있어서
            # (clamp/덧셈 순서에 따른 라운딩), 아래 EMA 수식을 타지 않고 그대로 반환한다.
            self._q_prev_j = torch.cat([q_L_j, q_R_j], dim=-1)
        q_prev_L_j = self._q_prev_j[..., : self.joint_dim]
        q_prev_R_j = self._q_prev_j[..., self.joint_dim : 2 * self.joint_dim]

        if not cfg.role_aware:
            alpha_hat = alpha_hat_shared.clamp(cfg.alpha_min, cfg.alpha_max)
            alpha_prev = alpha_hat if first_step else self._alpha_prev
            alpha = (1 - cfg.beta) * alpha_prev + cfg.beta * alpha_hat
            if first_step:
                q_tilde_L_j, q_tilde_R_j = q_L_j, q_R_j
            else:
                a = alpha[..., None]
                q_tilde_L_j = (1 - a) * q_prev_L_j + a * q_L_j
                q_tilde_R_j = (1 - a) * q_prev_R_j + a * q_R_j
            self._alpha_prev = alpha
            debug = {"rho": rho, "omega": omega_lr, "regime": regime, "alpha_hat": alpha_hat, "alpha": alpha}
        else:
            res_stab_j, res_act_j = self._resolve_role(res_L_j, res_R_j, stabilizer_is_left)
            omega_role = self._cosine_coop(res_stab_j, res_act_j, eps)
            alpha_hat_stab, alpha_hat_act = self._role_alpha_hat(alpha_hat_shared, omega_role)

            prev_stab = self._alpha_prev["stab"]
            prev_act = self._alpha_prev["act"]
            alpha_prev_stab = alpha_hat_stab if first_step or prev_stab is None else prev_stab
            alpha_prev_act = alpha_hat_act if first_step or prev_act is None else prev_act
            alpha_stab = (1 - cfg.beta) * alpha_prev_stab + cfg.beta * alpha_hat_stab
            alpha_act = (1 - cfg.beta) * alpha_prev_act + cfg.beta * alpha_hat_act

            if first_step:
                q_tilde_L_j, q_tilde_R_j = q_L_j, q_R_j
            else:
                q_stab_prev_j, q_act_prev_j = self._resolve_role(q_prev_L_j, q_prev_R_j, stabilizer_is_left)
                q_stab_j, q_act_j = self._resolve_role(q_L_j, q_R_j, stabilizer_is_left)
                a_stab, a_act = alpha_stab[..., None], alpha_act[..., None]
                q_tilde_stab_j = (1 - a_stab) * q_stab_prev_j + a_stab * q_stab_j
                q_tilde_act_j = (1 - a_act) * q_act_prev_j + a_act * q_act_j
                q_tilde_L_j, q_tilde_R_j = self._unresolve_role(q_tilde_stab_j, q_tilde_act_j, stabilizer_is_left)

            self._alpha_prev = {"stab": alpha_stab, "act": alpha_act}
            debug = {
                "rho": rho,
                "omega": omega_role,
                "regime": regime,
                "alpha_hat_stab": alpha_hat_stab,
                "alpha_hat_act": alpha_hat_act,
                "alpha_stab": alpha_stab,
                "alpha_act": alpha_act,
            }

        self._q_prev_j = torch.cat([q_tilde_L_j, q_tilde_R_j], dim=-1)
        q_tilde_L = torch.cat([q_tilde_L_j, q_L_g], dim=-1)
        q_tilde_R = torch.cat([q_tilde_R_j, q_R_g], dim=-1)
        return torch.cat([q_tilde_L, q_tilde_R], dim=-1), debug
