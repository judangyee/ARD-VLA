# LAC: Latent-Aware Controller 추론 필터 (Co-VLA, arXiv 2606.20285)

[← README로 돌아가기](../README.md)

Co-VLA의 Latent-Aware Controller(LAC)를 ARD-VLA에 이식한 **학습이 필요 없는 추론 시점
EMA 필터**입니다. SmolVLA(+ARD)가 매 서브태스크/실행 스텝마다 내놓는 액션의 "공유(베이스)
성분"과 "잔차(비대칭 보정) 성분"의 에너지 비율(ρ)과 두 팔의 협동 방향(ω)을 보고, 필터 강도
α를 상황별로 조절해서 떨림을 줄입니다. **이번 작업은 구조 구현 + 오프라인 합성 데이터
검증까지입니다** — 실제 SmolVLM2 가중치나 OpenArm 시뮬레이션에서는 돌려보지 않았습니다.

## ARD 구조가 LAC를 가능하게 하는 지점 (A안)

`VLAFlowMatching.denoise_step()`/`sample_actions()`는 **가산 구조**입니다:

```python
v_t = self.action_out_proj(suffix_out)          # 공유(베이스) 성분
if use_ard:
    v_t[..., :arm_dim]          += left_residual    # ARD 헤드가 만든 잔차(왼팔)
    v_t[..., arm_dim:2*arm_dim] += right_residual   # (오른팔)
```

`AsymmetricResidualHeads`의 마지막 레이어가 zero-init이라, 학습 초반에는 잔차가 정확히
0입니다(그래서 합성/미학습 모델로 테스트하면 ρ=0으로 항상 "macro" regime이 나옵니다 — 아래
"검증한 것" 참고). 이 가산 구조 덕분에 LAC가 필요한 "공유/잔차 분해"를 **추가 forward 없이**
얻을 수 있습니다: flow-matching denoise 루프(`num_steps`회)의 매 스텝에서 베이스 velocity와
잔차 velocity를 따로 적분하기만 하면 됩니다(선형 적분이라 `x_T = x_shared + x_residual`이
정확히 성립 — `VLAFlowMatching.sample_actions_with_role_split()`/`_denoise_loop()`의
`role_split_sink` side-channel 참고).

**RTC와는 함께 쓸 수 없습니다** — RTC가 여러 candidate velocity를 블렌딩해서 최종 `v_t`를
만들기 때문에, 이 선형 적분 전제가 깨집니다. `lac_mode != "none"`이면서
`rtc_config.enabled=True`면 config 생성 시점에 바로 에러가 납니다.

## LAC 수식

매 실행 스텝마다(그리퍼 채널 제외, 관절만):

```
E^s = ½(‖a^s_L‖ + ‖a^s_R‖),  E^r = ½(‖a^r_L‖ + ‖a^r_R‖)
ρ = E^r / (E^s + ε)
ω = −⟨a^r_L, a^r_R⟩ / (‖a^r_L‖‖a^r_R‖ + ε)
α̂ = α_base + Δ_macro   if ρ < τ_ρ          (두 팔이 같이 크게 움직임 — 평활화 강화)
     α_base + Δ_prec    elif ω > τ_ω        (반대 방향 협동 — 빠르게 통과)
     α_base − Δ_noise   else                (떨림 — 평활화 더 강화)
α̂ = clip(α̂, α_min, α_max)
α_t = (1−β)·α_{t−1} + β·α̂
q̃_t = (1−α_t)·q̃_{t−1} + α_t·q_t
```

첫 스텝은 `q̃_0 = q_1`(필터링 없이 그대로) — 부동소수점에서 `(1-a)*q+a*q`가 `q`와 비트
단위로 같지 않을 수 있어서, EMA 수식을 타지 않고 그대로 반환하도록 구현했습니다.

## 역할 인식(role_aware) 확장

Co-VLA 원본은 두 팔에 같은 α를 씁니다(`lac_mode="covla"`). ARD 확장(`lac_mode="role_aware"`)은
ARD의 Stabilizer/Actuator 역할별로 α를 따로 둡니다:
- **Stabilizer**: α 상한을 `lac_alpha_stab_max`(기본 0.5)로 낮춰서 항상 강하게 평활화.
- **Actuator**: Stabilizer/Actuator 잔차 사이의 ω가 `lac_tau_omega`를 넘으면(반대 방향
  협동, 예: 뚜껑 비틀기) `lac_alpha_act_cooperative`(기본 0.9)로 빠르게 통과시키고, 그
  외(macro/noise)에는 covla와 같은 추정치를 그대로 씁니다 — "떨림 판정일 때만 낮춤"이
  자동으로 성립합니다.
- 역할→팔 매핑은 `stabilizer_is_left`(기본 `ard_default_actuator_arm="right"`와 동일한
  전제, 즉 왼팔=Stabilizer) — role classifier 출력을 쓰려면 이 값을 그 출력으로 바꿔 넘기면
  되지만, **현재 ARD-VLA에는 그런 classifier가 없습니다**(폐루프 작업에서 추가한 Judge/
  Selector는 성공 판정/서브태스크 선택용이라 팔 역할과 무관).

## mirror_sign — 오른팔 관절 축 미러링 보정 (미검증)

OpenArm 좌우 팔은 일부 관절의 축 방향(또는 부모 프레임)이 미러링돼 있어서, 같은 물리적
포즈라도 왼팔/오른팔 관절각 부호가 반대로 나올 수 있습니다 — 보정 없이 ω를 계산하면
부호가 틀립니다. ARD-Gen의 `assets/openarm/openarm_bimanual.xml`에서 좌우 joint의
`axis`/`range` 속성을 직접 비교해서 얻은 기본 추정값:

| 관절 | mirror_sign | 근거 |
|---|---|---|
| joint1 | -1 | axis `0 1 0` vs `0 -1 0`, range 반전 |
| joint2 | -1 | axis 동일하지만 range 반전(부모 프레임 자체가 미러) |
| joint3 | +1 | axis·range 동일 |
| joint4 | +1 | axis·range 동일 |
| joint5 | +1 | axis·range 동일 |
| joint6 | -1 | axis `0 -1 0` vs `0 1 0` |
| joint7 | +1 | axis·range 동일 |

**이 값은 MJCF 텍스트 비교일 뿐입니다 — 실제 forward kinematics로 "왼팔 q, 오른팔 -q가
진짜 거울상 포즈"인지 시뮬레이션에서 확인하지 않았습니다(미검증).** `lac_mirror_sign`
config에 직접 넣어 쓰세요(길이는 `ard_arm_dim - ard_gripper_dim`과 같아야 함). 또한 현재
ARD-Gen의 `pipeline/to_lerobot.py`는 peg_in_hole/insertion 태스크에서 `action =
right_arm.action`만 내보냅니다(왼팔은 고정 그립) — 즉 지금 실제로 존재하는 데이터셋은
16-DoF 진짜 양팔 제어가 아니라 8-DoF 단일팔이라서, mirror_sign이 실제로 쓰일 양팔 제어
데이터로는 아직 검증할 수 없습니다.

## 설정 플래그

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `lac_mode` | `"none"` | `"none"`\|`"covla"`\|`"role_aware"`. `use_ard=True` 필요, RTC와 동시 사용 불가 |
| `lac_alpha_base` | `0.4` | regime 기본 α |
| `lac_delta_macro`/`lac_delta_prec`/`lac_delta_noise` | `0.4`/`0.3`/`0.1` | regime별 보정량 |
| `lac_tau_rho`/`lac_tau_omega` | `0.01`/`0.3` | regime 분리 임계값 |
| `lac_beta` | `0.2` | α의 시간적 평활화 |
| `lac_alpha_min`/`lac_alpha_max` | `0.05`/`0.95` | α 클립 범위 |
| `lac_epsilon` | `1e-6` | ρ/ω 분모 안정화 |
| `lac_alpha_stab_max` | `0.5` | role_aware: Stabilizer α 상한 |
| `lac_alpha_act_cooperative` | `0.9` | role_aware: 협동 시 Actuator α |
| `lac_mirror_sign` | `None`(전부 1.0) | 오른팔 관절 미러링 보정(관절 수 길이) |
| `lac_log` | `False` | 매 실행 스텝 rho/omega/alpha/regime을 `policy.lac_debug_log`에 기록 |

전부 `lac_mode="none"`이 기본이고, 그러면 기존 출력과 bit-for-bit 동일합니다
(`tests/test_ard_integration.py::test_lac_mode_none_matches_existing_select_action_output`).

## 어디에 연결했는지 — `select_action()`만

LAC는 `SmolVLAPolicy.select_action()`(청크에서 액션을 하나씩 꺼내 실행하는 경로)에만
연결했습니다 — `predict_action_chunk`/`predict_action_chunk_with_decisions`(closed_loop.py
러너가 쓰는 경로)는 청크를 통째로 반환하는 별도 경로라 이번 작업 범위에서는 건드리지
않았습니다. `select_action()`이 새 청크를 생성할 때(`lac_mode != "none"`이면)
`_get_action_chunk_with_lac()`가 ARD의 공유/잔차 분해를 받아 청크 전체(`n_action_steps`개)를
시간 순서대로 `LatentAwareController.step()`에 하나씩 넣어 필터링한 뒤 큐에 넣습니다 — 매
`select_action()` 호출마다가 아니라 **청크가 새로 만들어질 때 한 번에** 필터링합니다(수학적
으로는 동일 — 생성과 실행 사이에 새 정보가 들어오지 않는 순수 순차 재귀라서 — 하지만
`lac_log=True`로 로그를 보면 호출 횟수가 아니라 "청크당 n_action_steps개"로 쌓이는 걸
확인할 수 있습니다).

## 새/수정 파일

- `policies/smolvla/lac.py` (신규) — `LACConfig`, `LatentAwareController`.
- `configuration_smolvla.py` — `lac_*` 플래그 10개 + 검증(`use_ard` 필요, RTC 동시 사용
  금지, `lac_mirror_sign` 길이 검증).
- `modeling_smolvla.py` — `denoise_step()`의 `role_split_sink` side-channel(기존
  `last_collected_prefix_layers`와 같은 패턴), `_denoise_loop()`의 공유/잔차 적분,
  `sample_actions_with_role_split()`(신규, `sample_actions()`는 미변경),
  `SmolVLAPolicy.reset()`의 `self._lac` 생성, `_get_action_chunk_with_lac()`/
  `_apply_lac_filter()`(신규), `select_action()`의 분기.
- `scripts/compare_lac_modes.py` (신규) — 작은 합성 백본으로 none/covla/role_aware를
  비교하는 검증 스크립트(아래 "검증한 것" 참고).

## 검증한 것 / 못 한 것

**검증함** (`tests/test_lac.py`, `tests/test_ard_integration.py` — 전부 합성 모델로
오프라인 CPU에서 돈다):
- Co-VLA 손계산 케이스 3개(반대 비틀기→precision, 떨림→noise, 같이 이동→macro)가 수식
  그대로 나옴. α clip, β 평활화, `reset()`이 상태를 지우는지, 그리퍼 pass-through.
- role_aware: Stabilizer α가 상한을 안 넘는지, 협동 시 Actuator α가 높아지는지, 떨림일 때만
  낮아지는지, `stabilizer_is_left` 플래그 반영.
- mirror_sign이 ω의 부호를 실제로 바꾸는지, 길이 불일치 시 에러.
- config 검증: `lac_mode` 오탈자, `use_ard=False`와의 조합, RTC와의 조합, `mirror_sign` 길이.
- 실제 SmolVLAPolicy 전체: `sample_actions_with_role_split()`의 (공유+잔차) 분해가
  `sample_actions()`(미변경)와 bit-for-bit 동일한 `x_t`를 만들고 정확히 재구성되는지,
  `lac_mode="none"`이 기존과 bit-for-bit 동일한지, `select_action()`에서 첫 스텝은 통과
  하고 그 다음부터 실제로 필터링되는지, `lac_log`가 청크당 로그를 남기는지, `reset()`이
  LAC 상태를 재초기화하는지.
- `scripts/compare_lac_modes.py`로 none/covla/role_aware의 jerk(3차 차분)·ρ/ω/α 분포·
  regime 비율을 실제로 비교 실행 — covla/role_aware가 none보다 jerk를 낮추는 걸 확인했고,
  섭동 없이 돌리면(ARD 헤드 zero-init) 전부 "macro"로 결정론적으로 나오는 것과, role_aware가
  그 경우에도 Stabilizer α를 0.5로 캡하는 것(covla의 0.8과 다름)을 직접 봤다.

**못 함 (범위 밖, 구조/합성 데이터 검증만)**:
- 실제 SmolVLM2 가중치로 한 번도 안 돌려봄 — 전부 작은 합성 모델.
- mirror_sign 기본값은 MJCF 텍스트 비교일 뿐 forward kinematics로 검증 안 함, 게다가
  현재 ARD-Gen 데이터셋은 8-DoF 단일팔이라 실제 양팔 제어 데이터로 검증할 수조차 없음.
- `predict_action_chunk`/`predict_action_chunk_with_decisions`(closed_loop.py 러너)에는
  LAC를 연결하지 않음 — `select_action()`에만 연결(설계 결정, 위 참고).
- 실제 로봇/시뮬레이터에서 jerk 감소가 체감되는지, τ/Δ/β 등 하이퍼파라미터가 실제로
  적절한지는 전혀 실측하지 않음(Co-VLA 논문 기본값을 그대로 가져온 것).

```bash
pytest tests/test_lac.py tests/test_ard_integration.py
python scripts/compare_lac_modes.py
```
