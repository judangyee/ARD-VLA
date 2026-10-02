# Bridge Attention: 백본 여러 레이어를 조건으로 (`--use-bridge-attention`, VLA-Adapter식)

[← README로 돌아가기](../README.md)

기존 ARD head(`stabilizer_head`/`actuator_head`)는 action expert 트랜스포머의 **마지막 지점
출력**(`suffix_out`) 하나만 조건으로 받는다. VLA-Adapter(Wang et al., 2025)의 Bridge
Attention 아이디어를 빌려서, SmolLM2 백본의 **여러 중간 레이어**(초반/중반/후반/마지막 등
서로 다른 깊이) hidden state까지 cross-attention으로 동시에 조건받을 수 있게 확장했다.

- `lerobot/policies/smolvla/ard.py`(신규) — `BridgeAttention`: `suffix_out`을 쿼리로, 선택된
  여러 VLM 레이어의 prefix(이미지+언어) hidden state를 KV로 하는 멀티헤드 cross-attention.
  레이어마다 다른 깊이에서 왔다는 걸 구분하는 학습 가능한 `layer_embed`를 더한 뒤 이어붙여서
  KV로 쓴다. 출력은 `tanh(gate)`로 스케일되고 `gate`는 0으로 초기화되어(LLaMA-Adapter의
  zero-init attention과 같은 방식) 학습 초반에는 이 브랜치가 아무 영향도 주지 않는다 — 기존
  head의 마지막 레이어 zero-init과 합쳐져서, `--use-bridge-attention`을 켜도 학습 시작 시점의
  모델 출력은 꺼져 있을 때와 **완전히 동일**하다.
- `resolve_bridge_layer_indices()` — 조건으로 쓸 레이어 인덱스를 정한다. `--ard-bridge-layer-indices`를
  안 주면 (SmolVLA가 실제로 쓰는, 트림된) `num_vlm_layers`의 1/4·1/2·3/4·마지막 지점 4개를
  자동으로 고른다. **주의**: 이 인덱스는 원본 32개가 아니라 트림된(기본 16개) 레이어 기준이다
  — `layer_importance.py`가 분석하는 원본 인덱싱과 다르다.
- `smolvlm_with_expert.py`의 `SmolVLMWithExpertModel.forward(collect_layer_indices=...)` —
  VLM/action expert를 레이어 단위로 번갈아 처리하는 기존 인터리브 루프가 이미 매 레이어마다
  VLM 스트림의 중간 결과를 만들어내고 있어서, 지정한 레이어 인덱스를 지날 때 그 값을
  `self.last_collected_prefix_layers`에 캡처하기만 하면 된다 — **추가 forward pass 없이**
  공짜로 얻는다. `layer_importance.py`처럼 `text_model(...)`을 별도로 통째로 다시 돌리는
  방식은(그 방식도 처음엔 검토했다) 계산량이 그만큼 늘어나는 데다, 애초에 이 인터리브 루프는
  각 레이어의 `forward()`를 통으로 부르지 않고 `self_attn`/`mlp` 서브모듈을 직접 호출하는
  구조라 `layer_importance.py`가 쓰는 (레이어 전체에 거는) forward hook 방식 자체가 여기서는
  안 먹힌다 — 그래서 훅 대신 인터리브 루프 안에서 직접 캡처하는 방식을 택했다.
  추론(`sample_actions`)에서는 prefix KV 캐시를 만드는 시점에 **딱 한 번만** 계산되고, 이후
  `num_steps`번의 `denoise_step` 호출 전부가 `past_key_values`와 마찬가지로 그 결과를
  재사용한다(다시 계산하지 않음).
- `AsymmetricResidualHeads`에 `stabilizer_bridge`/`actuator_bridge`(각각 독립된 `BridgeAttention`
  인스턴스)를 추가했다 — Stabilizer/Actuator가 백본 특징 중 서로 다른 부분에 주목하도록 별도로
  학습된다. `SmolVLAConfig(use_bridge_attention=False)`가 기본값이라 켜지 않으면 기존과 100%
  동일하게 동작한다.

```bash
python scripts/train_ard.py --dataset-repo-id <...> --use-bridge-attention
# 조건 레이어를 직접 고르거나 헤드 수를 바꾸려면:
python scripts/train_ard.py --dataset-repo-id <...> --use-bridge-attention \
    --ard-bridge-layer-indices 4 8 12 15 --ard-bridge-num-heads 8
```

**파라미터 수 증가.** SmolLM2-360M의 공개적으로 알려진 config 값(hidden_size=960 — 이
샌드박스는 Hub 접근이 없어 직접 로드해서 재검증은 못 했다) 기준, `expert_width_multiplier=0.75`
(action expert hidden=720), `ard_arm_dim=7`, 기본 4개 조건 레이어로 계산하면:

| | 파라미터 수 | 전체(~450M) 대비 |
|---|---|---|
| ARD head만 (Bridge 없음) | 524,174 | 0.1165% |
| ARD head + Bridge Attention | 5,376,016 | 1.1947% |

즉 ARD head가 차지하는 비중이 **0.12% → 1.19%**로 늘어난다 — 절대량으로는 약 485만
파라미터(전체 모델의 약 1%) 추가로, 액션 전문가 트랜스포머 전체를 복제하는 것에 비하면 여전히
경량이다. `BridgeAttention` 인스턴스 하나(stabilizer 또는 actuator)의 내역은 `q_proj`/`out_proj`가
각각 `H_e × H_e`(720×720), `k_proj`/`v_proj`가 각각 `H_v × H_e`(960×720)이고, `layer_embed`는
`num_bridge_layers × H_v`(4×960)로 미미하다 — 이 계산은
`AsymmetricResidualHeads(use_bridge_attention=True, ...)`를 직접 만들어서 파라미터를 세는
방식으로 재현 가능하다.

**검증.** 이 샌드박스는 Hub/GPU 접근이 없어 실제 SmolVLM2 가중치로는 확인하지 못했다 — bitsandbytes
같은 CUDA 필수 요소는 이 기능에 없어서(순수 PyTorch `nn.Linear`/`scaled_dot_product_attention`만
사용), 이론상 CPU에서도 그대로 동작해야 하고 실제로 그렇게 확인했다. `tests/test_ard.py`에
`BridgeAttention`/`resolve_bridge_layer_indices`/`AsymmetricResidualHeads(use_bridge_attention=True)`
단위 테스트를 추가했고(zero-init 항등성, gradient 흐름, 잘못된 인자 거부, 하위호환 등), 별도로
소형 합성 SmolVLM 백본으로 `SmolVLAPolicy`를 실제로 만들어서: (1) 레이어 인덱스가 기대대로
계산되는지, (2) 학습 경로(`policy.forward`)에서 forward+backward 5스텝이 정상 동작하고
bridge 쪽에도 실제로 gradient가 흐르는지(첫 스텝엔 gate=0이라 정확히 0, 이후 스텝부터
0이 아니게 됨 — 설계대로), (3) 추론 경로(`sample_actions`)에서 prefix 레이어 수집이 정확히
1번만 일어나고 이후 `num_steps`번의 `denoise_step`이 전부 그 결과를 재사용하는지(직접 호출
횟수를 세어 확인), (4) `gradient_checkpointing_enable()`과 동시에 켜도 문제없는지까지
전부 확인했다. 실제 SmolVLM2 가중치로 손실이 실제로 더 잘 내려가는지(Bridge Attention의
효과 자체)는 실제 GPU 환경에서 로컬 데이터셋으로 학습해봐야 확인할 수 있다.

## Bridge Attention이 실제로 loss를 개선하는지 비교 (`scripts/compare_bridge_attention.py`)

`--use-bridge-attention`을 켠 모델과 끈 모델을 **같은 데이터 순서로** 순차 학습시켜서 loss
곡선을 직접 비교하는 스크립트입니다 (GPU 메모리를 하나만 쓰면서도 공정하게 비교하려고 동시가
아니라 순차로 돌되, 두 실행 모두 같은 시드로 `DataLoader`를 만들어 배치 순서를 똑같이 맞춥니다).

```bash
python scripts/compare_bridge_attention.py \
    --dataset-repo-id <HF_USER>/<DATASET> \
    --steps 300 \
    --batch-size 8
```

스텝별 loss 비교 표, `loss_curve.png`(두 곡선 겹쳐 그림), `loss_curve.csv`(원본 수치)를
`--output-dir`(기본 `outputs/compare_bridge_attention`)에 남깁니다. **중요한 과학적 주의점**:
이건 파라미터 개수를 맞춘 통제 실험이 아닙니다 — Bridge Attention을 켜면 ARD head의 학습 가능
파라미터 자체가 늘어나므로(위 "파라미터 수" 표 참고), loss가 더 잘 내려간다 해도 그게 "여러
레이어를 조건으로 주는 메커니즘" 덕분인지 "단순히 파라미터가 더 많아서"인지 이 비교만으로는
완전히 분리되지 않습니다 — 이 데이터셋에서 실제로 도움이 되는지/안 되는지의 1차 판단 용도로
쓰세요.

이 스크립트도 실제 데이터셋 접근이 없는 이 샌드박스에서는 끝까지 돌려보지 못했다 —
`train_ard.py`와 동일한 학습 루프 구조를 그대로 재사용했고(GradNorm은 비교 변수를 Bridge
Attention 하나로 좁히기 위해 뺐다), 이 스크립트에서 새로 추가된 부분(두 변형이 정확히 같은
배치 순서를 보도록 시드 고정하는 로직, 요약/CSV/그래프 저장 함수)은 합성 데이터로 직접
오프라인 검증했다. `--ard-symmetric` 플래그도 연결돼 있다([대칭 대조군 모드](ard.md#대칭-대조군-모드-ablation-ard_symmetric) 참고) — baseline/bridge 두 변형 모두 같은
`ard_symmetric` 값으로 돈다.
