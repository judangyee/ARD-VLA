# ARD: 비대칭 역할 분리 (Asymmetric Role Decomposition)

[← README로 돌아가기](../README.md)

ARD-VLA 연구계획서(양손 도구 조작 파인튜닝: 한 팔은 작업물을 고정하고, 다른 팔이 정밀한
조작을 수행)에 따라, SmolVLA의 액션 전문가(action expert)가 선택적인 비대칭 듀얼 헤드 모드를
지원하도록 확장했습니다. **Actuator 팔은 config로 고정되며, 항상 오른팔입니다** — 지시문마다
역할이 바뀌지 않습니다. `ard_default_actuator_arm`이 그 역할을 맡을 팔을 지정하고, 모든
샘플에 동일하게 적용됩니다.

**로봇 스펙(2026-10 확정, ARD-Gen).** 실제 타겟 로봇이 OpenArm + 공식 그리퍼로 확정되면서,
`ard_arm_dim` 기본값이 팔당 **7관절 + 그리퍼 1 = 8 DoF**로 바뀌었습니다(이전 임시값은 7).
`AsymmetricResidualHeads`/`compute_ard_losses` 등은 애초에 `arm_dim`을 순수 파라미터로만
받으므로(하드코딩된 7이 어디에도 없었음) 구조 변경 없이 기본값만 바뀐 것입니다 — bimanual
액션 벡터의 앞 8채널=왼팔, 다음 8채널=오른팔이라는 레이아웃 관례는 그대로입니다.

- `third_party/lerobot/src/lerobot/policies/smolvla/ard.py` (신규 파일) — `AsymmetricResidualHeads`
  (공유 flow-matching 출력 위에서 Stabilizer/Actuator 채널을 특화시키는 zero-init residual
  MLP), `resolve_actuator_is_first`(고정된 왼팔/오른팔 라우팅), `compute_ard_losses`
  (`alpha * L_stab + beta * L_act`, `L_stab = L_pos + λ·L_smooth`, `L_act = L_pos + λ·L_force +
  λ·L_traj` — 연구계획서의 손실 설계를 그대로 따름. `L_pos`는 별도의 L1 항을 다시 계산하지
  않고 베이스 모델 자체의 flow-matching 회귀 손실을 재사용합니다).

  **`L_smooth`/`L_traj`가 거는 대상 — velocity가 아니라 노이즈 제거된 액션 추정치.**
  `VLAFlowMatching.forward`는 flow-matching의 예측 velocity field `v_t`(목표는
  `u_t = noise - actions`)를 만든다. 초기 구현은 `L_smooth`(1차 차분)/`L_traj`(2차 차분)를
  이 `v_t` 자체에 걸었는데, 이건 개념적으로 틀렸다 — `u_t`는 매 타임스텝 독립적으로 샘플된
  `noise` 때문에 시간축으로 원래 거칠어서, `v_t`에 스무딩 벌점을 주면 "이 velocity는
  매끄러워야 한다"고 강요하는 셈이 되어 flow-matching 회귀 목표(`u_t`를 맞히는 것) 자체와
  정면으로 충돌한다. 지금은 직선 보간 `x_t = t·noise + (1-t)·actions`로부터 역산한
  `x0_hat = x_t - t·v_t`(노이즈 제거된 액션 추정치 — `v_t == u_t`일 때 `x0_hat`이 `actions`와
  정확히 같아짐을 대수적으로 보장)에 `L_smooth`/`L_traj`를 건다. `compute_ard_losses`의
  `stabilizer_traj_pred`/`actuator_traj_pred` 인자 이름도 이 의미를 명확히 하도록 바뀌었다.
- `configuration_smolvla.py` — 새 `use_ard`, `ard_arm_dim`, `ard_alpha`/`ard_beta`,
  `ard_lambda_{smooth,force,traj}`, `ard_default_actuator_arm`(기본값 `"right"`) 필드 추가,
  기본값은 전부 꺼짐 (`use_ard=False`이면 업스트림 SmolVLA와 완전히 동일하게 동작 — 플래그를
  켜지 않는 한 코드 경로가 전혀 바뀌지 않음을 검증함).
- `modeling_smolvla.py` — `VLAFlowMatching.forward`(학습 손실), `.sample_actions`/
  `.denoise_step`(추론, RTC 포함) 모두 동일한 residual head 보정과 고정 역할 라우팅을
  적용해서 학습과 추론이 서로 어긋나지 않도록 함.
- 선택적 샘플별 배치 키, 없으면 아무 영향 없음: `ard_force_target` (접촉력/토크 supervision —
  현재 이 레포의 어떤 데이터셋도 이 값을 제공하지 않아서, 데이터셋이 생기기 전까지 `L_force`는
  항상 0, 또는 `ard_use_force_head=True`면 [ForceHead](#forcehead-접촉력토크-전용-보조-헤드-ard_use_force_head)로 간다).

**검증.** 이 샌드박스의 네트워크 정책이 Hugging Face Hub를 막고 있고, `SmolVLAPolicy`는
`load_vlm_weights=False`여도 SmolVLM2 백본 config를 항상 다운로드해야 해서 — 여기서는
end-to-end로 생성해볼 수 없었습니다. 대신 `tests/test_ard.py`에서 오프라인으로 검증한
내용: `SmolVLAConfig`의 새 검증 로직, 고정 역할 라우팅이 항상 오른팔 채널을 Actuator head로
보내는지, 역할 split/combine 라운드트립, residual head의 zero-init/gradient 흐름,
`compute_ard_losses`의 수치 계산(이 과정에서 실제 버그도 하나 잡았습니다: `L_force`가
샘플별 타겟을 타임스텝별 예측값과 브로드캐스팅하려던 문제). `scripts/check_env.py`로는
기본(`use_ard=False`) 경로가 여전히 그대로 import/실행되는 것도 확인했습니다. 테스트는
`pytest`로 돌립니다:

```bash
python scripts/check_env.py
pytest tests/test_ard.py
```

`SmolVLAPolicy` 자체로 실제 forward/backward pass를 돌려보는 것(`use_ard=True`, 작은 VLM
차원으로)이 이 환경에 Hub 접근이 가능해지면 진행할 다음 검증 단계입니다.

## GradNorm으로 lambda 자동 조정 (`--use-gradnorm`)

`ard_lambda_smooth`/`force`/`traj`는 기본적으로 고정값(1.0)인데, `--use-gradnorm`을 주면
GradNorm(Chen et al., 2018)으로 매 스텝 자동 조정됩니다 (`lerobot/policies/smolvla/ard.py`의
`GradNormLambdas`). 핵심 아이디어: smooth/force/traj 세 항이 "공유 표현"(actuator/stabilizer
head 바로 직전의 `suffix_out` — 액션 전문가 트랜스포머 출력, `action_out_proj`와 `ard_heads`가
둘 다 이 텐서를 입력으로 받습니다)에 만드는 그래디언트 norm을 서로 균형 잡히게 맞춥니다.
초기 대비 유난히 느리게 줄어드는(=상대적으로 여전히 큰) 항일수록 그래디언트 norm 목표치를
크게 잡아서 해당 lambda가 커지도록 유도합니다.

```bash
python scripts/train_ard.py --dataset-repo-id <...> --use-gradnorm --gradnorm-alpha 1.5 --gradnorm-lr 0.025
```

**설계에서 중요한 점 두 가지**:
1. lambda(`GradNormLambdas.weights`)는 메인 total loss의 backward로 직접 업데이트되면 안
   됩니다 — 그러면 lambda가 그냥 0으로 수렴해버립니다(그래야 해당 항의 기여가 사라져서
   total이 작아지므로). 그래서 메인 loss를 만들 때는 항상 `lambda.detach()`를 쓰고, 진짜
   업데이트는 GradNorm 전용 손실(`L_grad`, `loss_dict['ard_grad_loss_tensor']`)로 학습
   루프가 별도 옵티마이저(`policy.model.ard_gradnorm.weights`만 대상으로)를 만들어 처리합니다.
   `policy.get_optim_params()`가 이 파라미터를 메인 옵티마이저에서 자동으로 제외합니다.
2. 순서가 중요합니다: `grad_loss.backward(inputs=[...], retain_graph=True)`와 메인
   `loss.backward()`를 **둘 다** 먼저 끝낸 뒤에야 `gradnorm_optimizer.step()` +
   `renormalize()`를 호출해야 합니다 — lambda를 먼저 in-place로 바꿔버리면 메인 loss의
   그래프가 그 값을 참조하고 있어서 "in-place로 바뀐 값" 에러가 납니다. `renormalize()`는
   GradNorm 논문대로 세 lambda의 합을 항상 3(태스크 개수)으로 재정규화합니다.

`force_target`이 없으면(이 레포의 모든 데이터셋이 그렇습니다) `force_loss`는 `suffix_out`과
연결되지 않은 상수 0이라 그 항의 그래디언트 norm은 항상 0입니다 — `lambda_force`는 사실상
갱신되지 않고(다른 두 lambda의 재정규화에 딸려서만 미세하게 움직임) 1.0 근처에 머뭅니다.
실질적으로는 smooth/traj 2-태스크 GradNorm이나 마찬가지입니다.

이 스크립트는 GPU/Hub 접근 없이 end-to-end로 못 돌려봤지만, `tests/test_ard.py`에
`GradNormLambdas`용 테스트 6개를 추가해서(초기 weights, 실제로 lambda가 움직이는지, 재정규화
후 합이 유지되는지, force처럼 그래프와 끊긴 항도 안 죽는지, `gradnorm=None`이면 기존 고정
lambda 경로와 완전히 같은지) 전부 통과를 확인했고, 별도로 **작은 합성 SmolVLM 백본**(진짜
Hub 다운로드 없이 `AutoConfig.from_pretrained`/`AutoProcessor.from_pretrained`만
몽키패치)으로 `SmolVLAPolicy.forward()` → `compute_ard_losses()` → GradNorm 업데이트까지
실제 코드 경로를 8스텝 돌려서 고정 lambda 방식과 비교했습니다:

```
고정 lambda=1.0:  ard_lambda_* 없음 (애초에 안 바뀜)
GradNorm 8스텝 후: lambda_smooth 1.00 -> 1.33   lambda_force 1.00 -> 1.00(거의 고정)   lambda_traj 1.00 -> 0.67
```

이 실험 당시(아래에서 설명하는 `x0_hat` 수정 전) `smooth_loss`/`traj_loss`가 `pos_loss`에
비해 원래 작았다는 사실과 별개로, GradNorm은 **그래디언트 norm**을 기준으로
판단하기 때문에 라벨 그대로의 손실 크기와는 다른 방향으로 조정될 수 있습니다 — 실제로 이
합성 백본 실험에서 `traj_loss`의 그래디언트 norm이 상대적으로 작게 나와서 GradNorm이
`lambda_traj`를 오히려 낮췄습니다. 8스텝만에 손실 자체의 비율(smooth/pos, traj/pos)은 고정
방식과 거의 같았는데, 이건 당연합니다 — GradNorm은 *미래* 그래디언트 업데이트 방향을
바꾸는 것이지, 그 순간의 손실값 자체를 바꾸는 게 아니라서 몇 스텝 만에 차이가 크게 벌어지진
않습니다. 이 실험은 8레이어짜리 무작위 초기화 장난감 백본 기준이라 절대적인 수치나 방향성이
진짜 SmolVLM2에서도 그대로 재현될지는 실제 GPU 환경에서 다시 확인이 필요합니다.

**주의 — 위 `smooth_loss`/`traj_loss` 스케일 수치는 수정 전(velocity 기반) 설계 기준이라
무효입니다.** `L_smooth`/`L_traj`가 `v_t`(velocity)가 아니라 `x0_hat`(노이즈 제거된 액션
추정치)에 걸리도록 고친 뒤, 같은 합성 백본으로 8스텝 probe를 수정 전/후 다시 돌려
비교했습니다:

```
[before-fix] (velocity v_t에 직접 건 경우)
step |        pos |     smooth |       traj | smooth/pos |   traj/pos
   1 |    2.68297 |    0.05490 |    0.29448 |      0.020 |      0.110
   4 |    2.54958 |    0.05027 |    0.21060 |      0.020 |      0.083
   8 |    2.32579 |    0.04368 |    0.09939 |      0.019 |      0.043

[after-fix] (x0_hat에 건 경우)
step |        pos |     smooth |       traj | smooth/pos |   traj/pos
   1 |    2.68297 |    1.02752 |    1.95705 |      0.383 |      0.729
   4 |    2.53317 |    1.07494 |    3.50039 |      0.424 |      1.382
   8 |    2.34088 |    0.98089 |    1.82520 |      0.419 |      0.780
```

(둘 다 8레이어가 아니라 2레이어짜리 무작위 초기화 장난감 SmolVLM 백본, seed=0, 동일한 배치
시퀀스 기준 — `pos` 열은 거의 그대로인데(ARD 외적인 베이스 flow-matching 손실이라 당연함),
`smooth`/`traj`는 수정 후 자릿수가 통째로 달라진다. 이건 버그였다는 증거이기도 하다 — 수정
전 `smooth_loss`/`traj_loss`는 `pos_loss`의 2~11%에 불과해 사실상 거의 기여를 못 했는데
(위 GradNorm 절의 "원래 작다"는 서술이 이 증상을 가리킨 것), noise가 타임스텝마다 독립
샘플이라 `v_t` 자체가 원래 거칠다는 걸 감안하면 오히려 더 커야 할 신호였다 — `x0_hat` 기반
으로 고친 뒤에는 `pos_loss`와 같은 자릿수(38~78% / 73~201%)로 커져서, `alpha`/`beta`/
`lambda_*` 조정이 실제로 의미 있게 작동할 수 있는 스케일이 됐다.)

**검증된 것**: `tests/test_ard.py`에 추가한 회귀 테스트로 — (1) `v_t == u_t`(완벽한 예측)일
때 `x0_hat`이 `noise`/`time`을 무엇으로 샘플하든 `actions`와 대수적으로 정확히 같아짐,
(2) 그 결과 `smooth_loss`/`traj_loss`가 `noise`와 무관하게 실제 `actions`의 1차/2차 차분과
정확히 같아짐, (3) 상수 궤적이면 완벽한 예측에서 두 손실이 정확히 0이 됨, (4) 수정 전 방식
(velocity를 직접 쓰는 대조군)과는 값이 달라짐 — 을 직접 수치로 확인했고, 기존 GradNorm
테스트 전부와 `force_loss` 로직(입력만 `x0_hat` 기반으로 바뀌었을 뿐 동작은 그대로)도 여전히
통과합니다.

**측정됨(실제로 문제가 될 수 있다는 쪽)**: `x0_hat = x_t - t·v_t`는 `t`(flow-matching
타임스텝)가 클수록 오차가 커질 수 있습니다 — `x0_hat - actions = -t·(v_t - u_t)`이므로
velocity 예측 오차가 `t` 배만큼 그대로 반영되고, 학습 초반/언더피팅 구간에서는 `t`가 1에
가까운(거의 순수 노이즈인) 샘플일수록 `v_t`의 예측 오차 자체가 실질적으로 더 큰 경향이
있습니다. 같은 장난감 백본을 8스텝 가볍게 학습시킨 뒤 `t`를 5개 구간으로 고정해서
`|x0_hat - actions|`의 평균/표준편차를 측정했습니다:

```
t 구간          | mean|x0_hat-actions| | std|x0_hat-actions|
[0.00, 0.20)    |                 0.12 |                0.07
[0.20, 0.40)    |                 0.36 |                0.07
[0.40, 0.60)    |                 0.60 |                0.08
[0.60, 0.80)    |                 0.83 |                0.10
[0.80, 1.00)    |                 1.06 |                0.10
```

평균 오차가 `t=[0, 0.2)`에서 `t=[0.8, 1.0)`까지 약 8.7배(0.12 → 1.06)로 뚜렷하게 커지는
걸 확인했습니다 — 우려가 실제 현상임을 이 장난감 백본 기준으로는 확인한 셈입니다. 다만 이게
진짜 SmolVLM2 규모에서도 학습에 실질적으로 해로운 수준인지(단순히 "큰 t의 샘플일수록
smooth/traj 신호가 더 시끄럽다" 정도인지, 아니면 학습을 실제로 방해하는지)는 확인되지
않았습니다 — 실제 GPU 환경에서 다시 측정해봐야 압니다.

### 시간 가중으로 완화하기 (`ard_reg_time_weighting="one_minus_t"`)

위 측정 결과에 대한 선택적 보정으로, `t`가 클수록 `x0_hat` 기반 smooth/traj 항의 기여를
줄이는 `(1-t)` 가중을 추가했습니다 — 기본값은 `"none"`(가중치 없음, 기존과 완전히 동일)
이고, 켤지 말지/가중 함수를 뭘로 할지는 실제 데이터로 확인한 뒤 사용자가 결정하는 게 낫다고
판단해 기본은 꺼둔 채로 남겨뒀습니다.

```bash
python scripts/train_ard.py --dataset-repo-id <...> --ard-reg-time-weighting one_minus_t
```

내부적으로 `VLAFlowMatching.forward`가 `ard_reg_time_weighting="one_minus_t"`일 때만
`1 - time`(배치의 flow-matching 타임스텝)을 계산해서 `compute_ard_losses`의
`reg_time_weights` 인자로 넘깁니다. `ard.py`의 `_reduce_reg_loss` 헬퍼가 이 가중치를
처리하는데, `reg_time_weights=None`(기본) 분기는 과거 코드의 `sq_diff.mean()`과
**bit-for-bit 동일**하도록 전혀 건드리지 않았습니다 — 가중치가 주어졌을 때만 샘플별 평균을
먼저 낸 뒤 가중 평균을 내는 별도 경로를 탑니다.

**검증.** `tests/test_ard.py`에서: `t=1`인 샘플만 있으면(가중치 0) 궤적이 아무리 거칠어도
smooth/traj_loss가 정확히 0이 되는지, `t=0`인 샘플만 있으면(가중치 1) 가중치 없는 경로와
근사적으로 같은 값이 나오는지, `reg_time_weights`를 아예 안 주면 과거 코드의 "전체 원소
`.mean()`" 식과 bit-for-bit 동일한지(`torch.equal`로 직접 비교) 확인했습니다. `config.
ard_reg_time_weighting`이 `"none"`/`"one_minus_t"` 외의 값을 거부하는 것도 확인했습니다.

## ForceHead: 접촉력/토크 전용 보조 헤드 (`ard_use_force_head`)

기존 `force_loss`는 `actuator_traj_pred[..., -1]`(actuator 블록의 마지막 "액션" 채널, 즉
관절 위치/속도 채널)을 힘의 대용값으로 재활용하고 있었는데, 이건 의미가 맞지 않습니다 — 그
채널은 실제로는 관절 채널이지 힘 센서 신호가 아닙니다. `ard_use_force_head=True`면
`suffix_out`에서 직접 스칼라 힘을 예측하는 별도 `ForceHead`(작은 MLP, `ard.py`)를 만들어서
`force_loss`를 거기서 계산합니다.

```bash
python scripts/train_ard.py --dataset-repo-id <...> --ard-use-force-head
```

- **출력**: `(batch, chunk_size)` — 액션 채널에 전혀 더해지지 않는 순수 보조(auxiliary)
  출력입니다. 추론 경로(`sample_actions`/`denoise_step`)에서는 아예 호출되지 않습니다 —
  `VLAFlowMatching.forward`(학습)에서만 쓰입니다.
- **초기화**: 기존 ARD head들과 달리 **zero-init을 쓰지 않습니다**. `AsymmetricResidualHeads`의
  zero-init은 "사전학습된 공유 projection 위에 처음엔 아무 영향도 주지 않는 항등으로
  시작한다"는 구조적 이유(그 출력이 `v_t`에 더해지기 때문)가 있지만, `ForceHead`의 출력은
  애초에 어디에도 더해지지 않는 독립 보조 출력이라 "항등"이라는 개념 자체가 없습니다. 오히려
  마지막 레이어를 0으로 초기화하면 `suffix_out`까지 역전파되는 그래디언트가 초기화 시점에
  정확히 0이 되어(zero 가중치 행렬을 통과하므로) GradNorm이 학습 첫 스텝부터 force 항의
  그래디언트 norm을 제대로 못 보는 문제가 생겨서, PyTorch 기본 초기화를 그대로 씁니다.
- **`force_target`은 있는데 `ard_use_force_head=False`면**: 과거처럼 엉뚱한 채널을 쓰지
  않고, `warnings.warn`으로 경고를 한 번 띄운 뒤 `force_loss=0`으로 처리합니다(기본 필터가
  동일 위치 경고를 프로세스당 한 번만 보여줘서, 매 스텝 호출돼도 자연스럽게 "한 번만"
  출력됩니다).
- **GradNorm과의 연결**: `ForceHead`의 입력이 정확히 `shared_activation`(=`suffix_out`)이라,
  `force_loss`가 `suffix_out`까지 역전파로 이어지는 계산 그래프를 자연스럽게 갖습니다 —
  `GradNormLambdas.compute_grad_loss`가 `torch.autograd.grad`로 이 그래디언트를 직접 재는
  방식이 정확히 설계대로 동작합니다.
- **옵티마이저/PEFT**: `get_optim_params()`는 `nn.Module` 파라미터 재귀 덕분에
  `ard_force_head`를 자동으로 포함합니다(별도 코드 불필요). `wrap_with_peft()`를 쓰는 비교
  스크립트(`profile_memory.py`/`compare_smolvla_ard.py`/`compare_bridge_attention.py`)들은
  ARD head를 다시 풀어주는 기존 수동 unfreeze 루프에 `ard_force_head`도 포함시켰습니다.

**검증.** `tests/test_ard.py`에서 `ForceHead` 출력 shape/gradient 흐름, `force_pred` 유무에
따른 경고/0-처리 동작, GradNorm이 `ForceHead` 경로로 `suffix_out`과 제대로 연결됨을
확인했습니다. `tests/test_ard_integration.py`(소형 합성 SmolVLM 백본으로 실제
`SmolVLAPolicy`를 만드는 통합 테스트)에서는 `ForceHead.forward`를 "호출되면 실패"로
바꿔치기한 뒤 `predict_action_chunk()`를 돌려서, 추론 경로에서 실제로 전혀 호출되지 않음을
직접 확인했습니다. 실제 SmolVLM2 가중치/실제 force 센서 데이터로는 아직 검증하지 못했습니다
— 이 레포의 어떤 데이터셋도 `ard_force_target`을 제공하지 않아서입니다.

## 그리퍼 손실 분리 (`ard_gripper_dim`)

**의견/근거.** 그리퍼는 "열림/닫힘"에 가까운 bang-bang성 신호라, 관절(joint) 채널처럼
부드럽게 이어지는 것이 정상이 아니라 오히려 빠르게 전환되는 것이 정상적인 동작입니다. ARD의
`L_smooth`(1차 차분)/`L_traj`(2차 차분)는 "예측 궤적이 매끄러워야 한다"는 벌점이라, 이걸
그리퍼 채널에 그대로 걸면 "그리퍼도 천천히 움직여라"라는 잘못된 신호를 주게 됩니다 — 사용자가
제안한 "그리퍼는 L_pos만, 관절은 기존 smooth/force/traj 유지"가 올바른 방향이라고 판단해 그
설계를 그대로 구현했습니다.

`ard_gripper_dim`(기본값 **1**)은 각 팔 `ard_arm_dim`개 채널 중 **마지막** `ard_gripper_dim`개를
그리퍼로 간주해서, smooth_loss/traj_loss 계산 전에 제외합니다(`ard.py`의 `_exclude_gripper`).
`L_pos`(베이스 flow-matching 회귀 손실)는 이 값과 무관하게 그리퍼 채널에도 그대로 걸립니다 —
즉 그리퍼는 "`L_pos`만 받고 smooth/traj 추가 벌점은 받지 않는다"는 설계입니다. `0`으로 주면
과거처럼 그리퍼도 smooth/traj에 포함됩니다(이전 세션들이 쓰던 동작과 동일).

```bash
python scripts/train_ard.py --dataset-repo-id <...> --ard-gripper-dim 1   # 기본값, 명시 예시
python scripts/train_ard.py --dataset-repo-id <...> --ard-gripper-dim 0   # 과거 동작으로 되돌리기
```

**검증.** `tests/test_ard.py`(`test_gripper_dim_excludes_last_channel_from_smooth_traj`,
`test_symmetric_losses_gripper_dim_excludes_last_channel`)에서: 관절 채널을 시간축으로 완전히
상수로 두고 그리퍼 채널만 무작위로 바꾼 합성 입력에 대해, `gripper_dim=0`이면 그 변동이
smooth/traj_loss에 그대로 반영되고 `gripper_dim=1`이면 정확히 0이 되는지 직접 확인했습니다
(`compute_ard_losses`/`compute_symmetric_losses` 양쪽 모두). 실제 그리퍼 개폐 패턴으로 학습
품질이 실제로 개선되는지는 실제 OpenArm 데이터로 학습해봐야 압니다.

## 관절 토크 입력 (`ard_use_joint_torque`)

**의견/근거.** 관절 토크를 모델에 어떻게 넣을지는 최소 두 가지 방향이 있습니다: (a) 기존
`observation.state` 벡터에 이어붙여서(concat) 이미 있는 범용 선형 projection(`state_proj`)이
그대로 처리하게 하거나, (b) 토크 전용 인코더/cross-attention 브랜치를 새로 만드는 것. 이번
구현은 (a) concat 방식을 택했습니다(사용자가 제안한 방향과 동일) — 이유는:

1. `state_proj`가 이미 `max_state_dim`(기본 32)까지 0으로 패딩된 입력을 범용으로 받고
   있어서, 토크를 그 남는 패딩 공간에 추가로 싣는 것만으로 **새 학습 파라미터가 전혀
   생기지 않습니다**(`state_proj`의 입력 폭은 항상 `max_state_dim`으로 고정이라, 내용만
   달라질 뿐 구조는 그대로입니다). 별도 브랜치를 만들면 그만큼 파라미터/메모리가 늘고,
   아직 실제 토크 데이터로 검증된 적 없는 신호에 그 비용을 투자할 근거가 약합니다.
2. 데이터셋/정규화 관점에서도 추가 작업이 없습니다 — `NormalizationMode.MEAN_STD`가 이미
   `observation.state` 전체에 채널별로 적용되므로, 토크 채널의 통계만 데이터셋이 제공하면
   자동으로 같은 방식으로 정규화됩니다.
3. 단점/트레이드오프: 토크는 관절 위치와 스케일/분포가 많이 다를 수 있는데, 이 방식은
   둘을 같은 선형 projection에 동등하게 맡깁니다 — 만약 실제로 이게 부족하다고 판명되면
   (예: 토크 신호가 묻힘), 토크 전용 작은 MLP를 먼저 통과시킨 뒤 concat하거나, 아예 별도
   cross-attention 브랜치(Bridge Attention과 비슷한 패턴)로 승격하는 걸 다음 단계로 검토할
   수 있습니다 — 이번 구현에는 포함하지 않았습니다(실제 토크 데이터 없이 미리 설계하는 건
   과도하다고 판단).

`ard_use_joint_torque=True`면 `SmolVLAPolicy.prepare_state()`가 배치의
`ard.ARD_JOINT_TORQUE`(`"observation.joint_torque"`) 키를 `observation.state` **뒤에**
concat한 뒤 `max_state_dim`까지 패딩합니다 — 학습(`forward`)과 추론
(`predict_action_chunk`/`sample_actions`) 양쪽 다 이 메서드 하나를 거치므로, 실제 배포 시에도
매 스텝 최신 토크 값이 그대로 입력됩니다. 키가 없으면(끄는 걸 잊었거나 데이터셋이 토크를 안
주는 경우) 조용히 무시하지 않고 `ValueError`를 던집니다. `force_target`/`ForceHead`와는 완전히
다른 개념입니다 — 이건 "모델이 보는 입력"이고, force_target/ForceHead는 "모델이 맞혀야 하는
보조 출력(정답)"입니다.

```bash
python scripts/train_ard.py --dataset-repo-id <...> --ard-use-joint-torque
```

**검증.** `tests/test_ard.py::test_joint_torque_config_validation`에서 `observation.state` +
토크 피처가 둘 다 `input_features`에 있을 때 `validate_features()`가 concat 후 차원이
`max_state_dim`을 넘는지 미리 걸러내는지 확인했습니다. `tests/test_ard_integration.py`
(소형 합성 SmolVLM 백본)에서: (1) 토크 값을 바꾸면 패딩 전 state 벡터의 해당 구간만 바뀌고
원래 state 채널은 그대로인지, (2) 키가 없으면 명확한 에러가 나는지, (3) 전체
forward+backward(학습)와 `predict_action_chunk`(추론) 경로가 끝까지 정상 동작하는지 확인했습니다.
실제 OpenArm 토크 센서 데이터로 학습 품질이 개선되는지는 검증하지 못했습니다 — 이 레포의
어떤 데이터셋도 아직 관절 토크를 제공하지 않습니다.

## Wrist 카메라 추가 (코드 변경 불필요)

SmolVLA의 이미지 입력 경로(`VLAFlowMatching.embed_prefix`/`prepare_images`)는
`self.config.image_features`(dict)를 그냥 순회해서 처리하므로, 카메라가 몇 개든 이름이
무엇이든 **코드 변경 없이** 그대로 들어갑니다 — `configuration_smolvla.py`의 `empty_cameras`
주석도 원래 "left and right wrist cameras in addition to the top camera"를 언급하고
있었습니다. 실제 학습(`scripts/train_ard.py`)은 `LeRobotDataset`의 피처를 그대로
`input_features`로 쓰므로, 데이터셋에 `observation.images.wrist_left`/`wrist_right`같은 키만
있으면 wrist 카메라가 추가 이미지 입력으로 자동으로 들어갑니다.

실질적인 "config 확장"은 이 레포의 합성 벤치마크/비교 스크립트(`profile_memory.py`,
`compare_smolvla_ard.py`, `scripts/verify_with_real_weights.py`)들이 쓰던 익명 카메라 이름
(`cam0`, `cam1`, ...)을 `top`/`wrist_left`/`wrist_right`로 바꾼 것입니다 — 기능적으로는
동일하지만, wrist 카메라가 실제로 들어간다는 걸 스크립트 출력/코드에서 바로 알 수 있게
했습니다([프로파일링 도구 문서](profiling_tools.md) 참고).

## 대칭 대조군 모드 (ablation, `ard_symmetric`)

ARD의 핵심 가정("Stabilizer는 smooth만, Actuator는 force+traj만")이 실제로 도움이 되는지
공정하게 비교하기 위한 대조군 모드입니다. 켜면 head 구조/파라미터 수는 ARD와 완전히
동일하게 유지한 채(`AsymmetricResidualHeads`를 그대로 재사용), 손실 **결합 방식**만
바뀝니다 — 양팔 모두 `L_pos + lambda_smooth·L_smooth + lambda_traj·L_traj`를 동일하게
적용하고(ARD처럼 stabilizer=smooth만/actuator=force+traj만으로 쪼개지 않음), `alpha`/`beta`는
항상 0.5/0.5로 강제됩니다(사용자가 다른 값을 줬으면 경고 후 덮어씁니다).

```bash
python scripts/train_ard.py --dataset-repo-id <...> --ard-symmetric
```

`ard.py`의 `compute_symmetric_losses`가 담당하며, 기존 `compute_ard_losses`는 전혀 건드리지
않았습니다(완전히 분리된 새 함수) — `ard_symmetric=False`(기본)일 때의 동작은 리팩토링 전과
동일한 인자로 `compute_ard_losses`를 호출하므로 bit-for-bit 그대로입니다.

**force 항은 대칭화하지 않았습니다.** smooth/traj(궤적의 일반적인 매끄러움)는 양팔 모두에
자연스럽게 의미가 있는 것과 달리, 접촉력/토크는 도구를 조작하는 Actuator 팔에만 물리적으로
의미가 있는 신호입니다 — 작업물을 가만히 붙잡고 있는 Stabilizer 팔에는 애초에 "힘을
추적한다"는 과제 자체가 성립하지 않습니다. 그래서 `force_loss`는 ARD 모드와 동일하게
actuator 쪽에만 유지했습니다. 이건 판단이 갈릴 수 있는 지점이라 여기 명시해둡니다.

### 세 변형 비교 (`scripts/compare_smolvla_ard.py --variants base,ard,symmetric`)

`--variants`를 쉼표로 구분된 목록으로 줘서 base(기본 SmolVLA, 단일팔 7 DoF) / ard(비대칭,
bimanual 14 DoF) / symmetric(대칭 대조군, bimanual 14 DoF) 중 원하는 조합만 비교할 수
있습니다:

```bash
!python scripts/compare_smolvla_ard.py --variants base,ard,symmetric
```

기본값은 `"base,ard"`(이 플래그가 추가되기 전과 동일하게 동작) — `symmetric`은 새 기능이라
기본에는 포함되지 않습니다. `ard`와 `symmetric` 변형은 같은 `AsymmetricResidualHeads`를
쓰므로 전체/학습 가능 파라미터 수가 정확히 같게 나옵니다(공정 비교 목적).
`train_ard.py`/`compare_bridge_attention.py`에도 `--ard-symmetric` 플래그가 연결돼 있습니다.

**검증.** `tests/test_ard.py`에서 config 검증(대칭 모드 강제 alpha/beta), 양팔 손실 "형태"가
동일함을 입력을 맞바꿔서 `stabilizer_loss`/`actuator_loss`가 정확히 맞바꿔지는 것으로 확인,
파라미터 수 동일성(같은 `AsymmetricResidualHeads` 클래스를 두 번 만들어 비교), force 처리가
ARD 모드와 동일함을 확인했습니다. 실제 GPU에서 세 변형을 비교 학습해본 적은 없습니다.
