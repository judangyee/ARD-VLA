# 파라미터/메모리 프로파일링 도구

[← README로 돌아가기](../README.md)

학습 루프 밖에서 모델 구성·메모리·속도를 미리 들여다보기 위한 스크립트 모음입니다.

## 파라미터 구성 확인 (`scripts/count_params.py`)

SmolVLA(+ARD) 전체 파라미터를 비전 인코더 / LLM(SmolLM2) / Action Expert / ARD head /
나머지 shim 레이어로 나눠서 개수와 비중을 보여주고, 지정한 해상도별로 이미지가 실제로 몇 개의
토큰이 되는지도 출력합니다. `load_vlm_weights=False`라 전체 가중치를 받지는 않지만, config는
Hugging Face Hub에서 받아야 해서 이 샌드박스에서는 실행이 안 됩니다 (compile/import/CLI 파싱은
확인했고, 실행하면 예상대로 네트워크 호출 단계에서 막히는 것까지 확인했습니다).

```bash
python scripts/count_params.py --resolutions 384 512 768
```

중요: SmolVLA는 SmolLM2-360M의 레이어를 전부 쓰지 않고 `config.num_vlm_layers`(기본값 16)개만
물리적으로 잘라서 씁니다(`smolvlm_with_expert.py`의 `text_model.layers = ...[:num_vlm_layers]`).
이 스크립트는 원본 레이어 수와 실제 사용하는 레이어 수를 둘 다 보여줘서 이 부분을 헷갈리지
않게 합니다.

## Gradient checkpointing (`SmolVLMWithExpertModel.gradient_checkpointing_enable()`)

`SmolVLMWithExpertModel.forward()`는 VLM과 action expert를 레이어 단위로 번갈아 호출하는
커스텀 루프라서, transformers의 표준 `model.gradient_checkpointing_enable()` 훅이 걸리지
않습니다 (그 훅은 서브모듈의 표준 `forward()` 호출 경로를 가로채는데, 이 루프는 그 경로를
쓰지 않습니다). 그래서 이번에 별도로 추가했습니다:

```python
vlm_expert = policy.model.vlm_with_expert
vlm_expert.gradient_checkpointing_enable()   # 켜기
vlm_expert.gradient_checkpointing_disable()  # 끄기
```

내부적으로 레이어 하나의 본문을 `_run_layer()`로 뽑아내고, 학습 중(`self.training`)이면서
KV 캐시를 안 쓰는 forward 경로(`use_cache=False`, `fill_kv_cache=False` — 즉 학습 시
`VLAFlowMatching.forward`가 실제로 쓰는 경로)에서만 `torch.utils.checkpoint.checkpoint(
self._run_layer, ..., use_reentrant=False)`로 감쌉니다. 추론(`sample_actions`/`denoise_step`,
KV 캐시 사용)에는 영향이 없습니다.

이 기능은 실제 SmolVLM2 모델로 검증하지 못했습니다(Hub 접근 차단, GPU 없음) — 대신 동일한
호출 시그니처(텐서 리스트 + `None` + non-tensor 인자가 섞인 형태)를 흉내 낸 가짜 레이어로
체크포인팅 유무에 따라 forward 출력과 gradient가 정확히 일치하는지 별도로 검증했습니다.
`check_env.py`/`tests/test_ard.py`는 이 플래그가 기본 `False`라 회귀 없이 통과합니다.

## 메모리 프로파일링 (`scripts/profile_memory.py`)

LoRA + bf16 autocast + gradient checkpointing을 모두 켠 상태에서, bimanual 액션
(16 DoF = 팔당 8 DoF, OpenArm+그리퍼 확정 스펙, `chunk_size=50`) 더미 배치로 forward+backward를
한 번 돌려 배치 사이즈별
(`1, 2, 4, 8, 16, 32` 기본값) `torch.cuda.max_memory_allocated()` 최대 메모리를 표로 출력하는
스크립트입니다. Colab/Kaggle 노트북에서 GPU 런타임으로 바로 돌릴 수 있게 단일 파일로
작성했습니다:

```bash
!pip install -e "third_party/lerobot[smolvla,peft]"
!python scripts/profile_memory.py
```

`--no-lora`, `--no-bf16`, `--no-grad-checkpoint`, `--no-ard`로 각 기법을 개별적으로
끄고 비교할 수 있고, 배치 사이즈 도중 OOM이 나도 스크립트가 죽지 않고 해당 칸을 "OOM"으로
표시한 뒤 나머지 배치 사이즈를 계속 시도합니다.

LoRA는 기존에 있던 `PreTrainedPolicy.wrap_with_peft()`를 그대로 사용합니다 — 다만
SmolVLA의 기본 LoRA 타겟(`lm_expert`의 attention projection들)에는 ARD의
`stabilizer_head`/`actuator_head`(+`ard_force_head`)가 포함되지 않아서, `wrap_with_peft()`가
나머지 전부를 얼린 뒤 이 헤드들을 명시적으로 다시 `requires_grad_(True)`로 풀어줍니다
(그렇지 않으면 ARD head가 통째로 학습에서 빠집니다).

기본 실행은 SmolVLA의 레이어 프루닝(`num_vlm_layers`로 SmolLM2를 앞쪽 몇 개 레이어만 쓰도록
자르는 것) 적용 여부를 **둘 다** 프로파일링해서 표를 두 개 냅니다 — "적용 O"는
`--num-vlm-layers`(기본 16)로 자른 기본 SmolVLA 설정, "적용 X"는 원본 SmolLM2 레이어 수를
그대로 쓰는 모델입니다. 원본 레이어 수 쪽은 `num_expert_layers`가 기본 `-1`이라 action
expert도 VLM 레이어 수를 따라가며 같이 커지므로 메모리를 훨씬 많이 쓰고 더 빨리 OOM이 날 수
있습니다. `--layer-pruning-mode pruned` 또는 `unpruned`를 주면 그중 하나만 돌려서 시간을
아낄 수 있습니다.

`--vlm-layer-indices`를 주면 "앞쪽 N개"라는 기본 규칙 대신 임의의 원본 레이어 인덱스 조합을
그대로 써서 ARD-VLA를 빌드하고, "기본(pruned)" vs "사용자 지정(custom)" 두 표를 비교
출력합니다 (이때는 `--layer-pruning-mode`가 무시됩니다). `scripts/layer_importance.py`가
코사인 유사도 기준으로 골라준 레이어들을 그대로 넣어서 실제로 문제없이 빌드/학습되는지
확인하는 용도입니다 — 레이어 개수가 같으면(기본 16개) 메모리 자체는 어차피 거의 동일하게
나올 걸로 예상됩니다(레이어들이 전부 동형 구조라 메모리는 "몇 개냐"로 결정되지 "어떤
인덱스냐"와는 무관하기 때문). 예:

```bash
!python scripts/layer_importance.py   # 레이어별 중요도 순위 확인
!python scripts/profile_memory.py --vlm-layer-indices 2 3 4 7 8 9 10 11 12 14 15 16 20 21 25 26
```

이 스크립트는 실제 Colab GPU 런타임에서 검증했습니다. 처음 실행할 때 두 가지 문제가
나올 수 있는데(둘 다 이 레포/스크립트 버그는 아니고 Colab 환경 특성입니다):
- `pip install -e ...`가 torch/torchvision을 재설치하면서 "You must restart the runtime"
  경고가 뜨면, 재시작 후 새 셀에서 `%cd`부터 다시 하고 스크립트만 실행하세요 (설치를 다시 할
  필요는 없습니다).
- `wrap_with_peft()`가 peft의 `dispatch_torchao` 단계에서 `ImportError: Found an
  incompatible version of torchao`를 던지면, Colab에 미리 깔린 `torchao`가 peft 요구
  버전과 안 맞아서입니다 — 저희는 양자화를 안 쓰니 `!pip uninstall -y torchao`로 지우고
  다시 실행하면 됩니다.

## 기본 SmolVLA vs ARD-VLA 비교 (`scripts/compare_smolvla_ard.py`)

`profile_memory.py`와 같은 패턴(LoRA + bf16 + gradient checkpointing 기본 켜짐)으로, 이번엔
"레이어 프루닝 O/X"가 아니라 **모델별**로 같은 배치 사이즈(`1, 4, 16, 32` 기본값)로 비교합니다:

- **base**: 단일팔 8 DoF, `use_ard=False` — 원본 액션 헤드만 사용
- **ard**: bimanual 16 DoF, `use_ard=True` — `AsymmetricResidualHeads` 적용(비대칭)
- **symmetric**: bimanual 16 DoF, `use_ard=True, ard_symmetric=True` — 대칭 대조군
  (자세한 건 [ARD 문서의 "대칭 대조군 모드"](ard.md#대칭-대조군-모드-ablation-ard_symmetric) 참고)

```bash
!python scripts/compare_smolvla_ard.py --variants base,ard
# 세 변형 모두 비교하려면:
!python scripts/compare_smolvla_ard.py --variants base,ard,symmetric
```

출력은 표 두 개입니다: (1) 변형별 전체 파라미터 수 / LoRA 학습 대상 파라미터 수, (2) 배치
사이즈별 각 변형의 peak memory(GB)와 forward pass 시간(ms)을 나란히 놓은 비교표. 메모리는
`profile_memory.py`와 동일하게 forward+backward 기준(실제 학습 스텝의 메모리 최고점을 반영),
시간은 backward를 뺀 forward 단독 기준입니다 — 이 둘을 같은 배치에서 한 번에 재느라, 시간
쪽엔 별도 warmup이 없어서 첫 호출(특히 batch_size=1)은 CUDA 커널 초기화 비용이 섞여 다소
부풀려질 수 있습니다.

`--variants`에 없는 모델은 그냥 돌지 않습니다. 나머지 옵션(`--no-lora`, `--no-bf16`,
`--no-grad-checkpoint`, `--lora-r`/`--lora-alpha`, `--batch-sizes`, `--ard-symmetric` 등)은
`profile_memory.py`와 동일하게 동작합니다.

이 스크립트는 `profile_memory.py`와 같은 검증된 패턴을 그대로 재사용했지만, 스크립트 자체를
실제 GPU에서 돌려보지는 못했습니다 — `py_compile`/CLI 파싱 확인 외에, 표 출력 로직(OOM 셀
처리 포함)은 가짜 데이터로 직접 검증했습니다.

## 레이어 중요도 분석 (`scripts/layer_importance.py`)

SmolVLA는 SmolLM2 백본(보통 원본 32레이어)에서 `config.num_vlm_layers`(기본 16)개만 남기고
쓰는데, 그 선택은 `smolvlm_with_expert.py`의 `text_model.layers[:num_vlm_layers]` — 그냥
**앞쪽 절반만 남기고 뒤쪽을 통째로 버리는** 슬라이싱입니다. 이 스크립트는 "정말 뒤쪽 16개가
가장 안 중요한 레이어가 맞는지"를 직접 측정해서 확인합니다.

방법은 ShortGPT류 레이어 중복성 분석과 같습니다: 더미 이미지+언어 토큰으로 SmolVLA가 실제
쓰는 `embed_image()`/`embed_language_tokens()`를 통해 대표 시퀀스를 만들고, 원본(트림 안 한)
SmolLM2 전체 레이어에 forward hook을 걸어 레이어별 입력/출력 hidden state의 코사인 유사도를
잰 뒤 `중요도 점수 = 1 - 평균 코사인 유사도`로 순위를 매깁니다 (유사도가 1에 가까울수록 그
레이어는 입력을 거의 안 바꾼다는 뜻이라 중요도가 낮음).

```bash
!python scripts/layer_importance.py
```

출력은 (1) 전체 레이어 중요도 순위표, (2) "코사인 유사도 기준 중요도 하위 N개(N=SmolVLA가
실제로 버리는 레이어 수)"와 "SmolVLA가 실제로 스킵 중인 레이어" 두 목록의 overlap 비율 및
차이 나는 레이어 목록입니다.

이 스크립트도 GPU/Hub 접근이 없는 이 샌드박스에서 실행은 못 해봤습니다 — 다만 핵심 로직(forward
hook으로 레이어 입출력을 뽑는 부분, 중요도 계산, overlap 비교)은 실제 `transformers` 라이브러리의
`LlamaDecoderLayer`(SmolLM2가 쓰는 것과 같은 클래스) 소스를 직접 읽고 시그니처를 맞췄고, 그
호출 관례(`GradientCheckpointingLayer.__call__`이 `super().__call__()`으로 위임해서 forward
hook이 정상 동작하는 것, 레이어가 튜플이 아니라 텐서를 그대로 반환하는 것)를 그대로 흉내 낸
가짜 레이어 스택으로 hook 캡처 → 점수 계산 → 순위/overlap 로직까지 전부 직접 검증했습니다
(direction-flip 레이어는 중요도가 높게, 항등에 가까운 레이어는 낮게 나오는 것 확인). 실제
SmolVLM2 모델의 `text_model` 클래스가 다른 시그니처를 쓸 가능성만 실행 전까지 확신할 수 없습니다.

여기서 나온 레이어 조합을 실제로 ARD-VLA에 적용해보려면(예: "앞쪽 16개" 대신 이 스크립트가
추천한 16개), `SmolVLAConfig(vlm_layer_indices=[...])`를 쓰면 됩니다 — `num_vlm_layers`가
"앞에서부터 N개"만 고정으로 자르는 것과 달리, `vlm_layer_indices`는 원본 레이어 중 임의의
인덱스 조합을 그대로 선택합니다(깊이 순서 보존을 위해 내부적으로 오름차순 정렬해서 사용).
`scripts/profile_memory.py --vlm-layer-indices ...`로 바로 프로파일링해볼 수 있습니다
(자세한 건 위 "메모리 프로파일링" 절 참고).
