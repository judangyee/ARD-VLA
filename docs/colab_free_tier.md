# 무료 Colab에서 RoCo 데이터로 스모크/베이스라인 학습 돌리기

[← README로 돌아가기](../README.md)

`scripts/colab/run_colab_training.py`가 진입점이다. 목표는 "무료 Colab(GPU 종류
보장 없음, 세션이 예고 없이 끊김, 백그라운드 실행 불가)에서도 RoCo 데이터로 작은
학습을 돌릴 수 있는 환경" — 실제 대규모 학습이 아니라 **파이프라인 전체가 깨지지
않는지, 그리고 끊겨도 이어갈 수 있는지**를 확인하는 수준이다.

## 0. 실행 순서

### 0-1. Colab Secrets로 토큰 준비

좌측 사이드바의 🔑(Secrets) 메뉴에서 미리 등록해둔다 — **노트북 코드에 토큰을 직접
문자열로 적지 말 것**(깃에 커밋되거나 노트북을 공유하면 그대로 노출된다):

- `HF_TOKEN`: `rocochallenge2025/rocochallenge2026_Industrial_Assembly`가 비공개/게이트된
  데이터셋이면 필요하다(공개 데이터셋이면 생략 가능 — 어느 쪽인지는 이 샌드박스에서
  확인하지 못했다, 미확인).
- `GITHUB_TOKEN`: 이 저장소(`judangyee/ard-vla`)가 비공개면 클론에 필요하다.

```python
from google.colab import userdata
import os

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
github_token = userdata.get("GITHUB_TOKEN")  # 클론 명령에만 쓰고 환경변수로 남기지 않아도 됨
```

### 0-2. Drive 마운트 + 저장소 클론 + 의존성 설치

```python
from google.colab import drive
drive.mount("/content/drive")

!git clone https://{github_token}@github.com/judangyee/ard-vla.git /content/ard-vla
%cd /content/ard-vla
!git checkout configurable-ard   # 또는 실제 작업 브랜치

!pip install -q -e third_party/lerobot
!pip install -q huggingface_hub pyyaml
```

### 0-3. 스모크 먼저 (권장)

실제 데이터/가중치 없이 전체 파이프라인(GPU 감지 → 스키마 검증 → 정규화 통계 → 학습
→ 체크포인트 재시작 → 오프라인 평가)이 이 Colab 인스턴스에서 안 깨지는지 1분 내로
확인한다:

```bash
python scripts/colab/run_colab_training.py --smoke --smoke-steps 60
```

화면 맨 아래 `=== --smoke 결과 요약 ===`에 각 단계 PASS/FAIL과 스텝당 시간/GPU 메모리
최댓값이 한 번에 나온다. 하나라도 FAIL이면 실제 학습을 시작하지 않는 게 좋다.

### 0-4. 실제 실행

```bash
python scripts/colab/run_colab_training.py \
    --drive-root /content/drive/MyDrive/ard_vla_colab \
    --steps 2000
```

`--drive-root`는 **반드시 Drive 경로**를 줘야 한다 — 데이터셋 캐시/에피소드 분할
파일/체크포인트가 전부 여기 쌓이고, 로컬 디스크(`/content`)에 두면 세션이 끊기는
순간 전부 사라진다.

## 1. 끊겼을 때 이어하는 방법

Colab이 끊기면(또는 런타임을 수동으로 재시작하면): **0-2(클론/설치)부터 다시 하고,
0-4의 명령을 정확히 같은 인자로 다시 실행**한다.

```bash
python scripts/colab/run_colab_training.py \
    --drive-root /content/drive/MyDrive/ard_vla_colab \
    --steps 2000
```

- `--drive-root` 아래 `outputs/checkpoints/`에서 가장 최신 체크포인트를 자동으로
  찾아 모델/옵티마이저/스텝/RNG 상태를 복원하고, 그 지점부터 이어서 학습한다.
- **`--steps`/`--split-seed`/`--data-seed`/`--batch-size`는 재시작할 때마다 똑같이
  줘야 한다** — 다르게 주면 스케줄러가 엉뚱한 진행률을 계산하거나(`--steps`),
  데이터 분할/순서가 바뀐다(나머지). `--drive-root/episode_split.json`이 저장돼
  있어서 `--split-seed`를 다르게 줘도 기존 분할 파일을 그대로 쓰지만(에피소드 수가
  같다는 전제 하에), `--steps`/`--batch-size`는 저장되지 않으므로 사람이 직접
  기억해서 맞춰야 한다 — 알려진 한계(아래 참고).
- 데이터셋은 `--drive-root/hf_cache`에 캐시되므로 재다운로드하지 않는다(파일 단위
  콘텐츠 주소 캐시 — 받던 중 끊긴 파일 하나만 다시 받으면 됨).

## 2. 알려진 한계 (정직하게 표시)

이 레포의 샌드박스는 GPU/Hugging Face Hub 접근이 모두 막혀 있어, 아래 항목은
**로직 자체는 몽키패치로 단위 테스트했지만 실제 GPU/실제 데이터셋으로 끝까지
실행해본 적이 없다**(미확인). Colab에서 처음 돌릴 때 특히 주의해서 볼 지점:

| 항목 | 상태 |
|---|---|
| bf16/fp16 분기 로직(`env_setup.py`) | `torch.cuda.is_bf16_supported(including_emulation=False)`로 T4/A100을 가르는 로직은 몽키패치로 검증됨. **실제 T4 위에서 돌려본 적 없음** — fp16 autocast+GradScaler 조합이 SmolVLA 규모에서 실제로 안정적인지(inf/nan 없이)는 미검증. |
| 배치 크기 자동 추정(`_AUTO_BATCH_SIZE_TABLE`) | VRAM만 보고 추정한 값이고 **실측 없음**. 카메라 3대(240×320) + chunk_size=20 조합에서 실제 OOM 경계가 어딘지 전혀 모른다 — `--batch-size 1`로 시작해서 점점 올려보는 걸 권장. |
| 데이터셋 전체 용량 조회(`dataset_fetch.query_dataset_size`) | `HfApi.dataset_info(files_metadata=True)` 호출 자체가 이 샌드박스에서 실행 불가(네트워크 차단) — API 시그니처만 보고 작성, 실제 응답 형태(일부 파일 size가 비는 경우 등)는 미확인. |
| 영상 인코딩(`use_videos=True`, 기본 코덱 `libsvtav1`) | 이 샌드박스에 `ffmpeg` 자체가 없다(확인됨, `which ffmpeg` 실패) — Colab 기본 이미지엔 보통 ffmpeg가 있지만 `libsvtav1` 인코더가 포함돼 있는지는 미확인. 데이터셋을 직접 만드는 게 아니라 **받기만** 하므로 이 스크립트 경로에서는 영향이 제한적이지만, LeRobotDataset 내부가 비디오 디코딩에 어떤 backend를 쓰는지도 미확인. |
| `schema_check.py`의 이름 판정 | 정확한 문자열이 아니라 "left/right"·"ee/pos/quat"·"joint"·"gripper" 부분 문자열 휴리스틱이다(모듈 docstring 참고) — 실제 RoCo info.json이 이 휴리스틱을 통과하는지 **미확인**. 처음 실행할 때 반드시 로그에 찍히는 실제 채널 이름을 눈으로 한 번 더 확인할 것. |
| 체크포인트 재시작의 "데이터 순서" | `BatchSource`가 에포크를 `data_seed + epoch`로 재시드하고, 재시작 시 그 지점까지 배치를 "다시 읽고 버려서" 되돌린다(`batch_source.py`/`checkpoint.py` docstring) — 완전한 무비용 재개가 아니라, 재시작 지점이 늦을수록(에포크 후반) 그만큼 느리다. |
| 오프라인 평가의 `prev_action` 근사 | 평가 에피소드마다 **중간 지점** 청크 하나만 대표로 쓴다(에피소드 전체를 다 평가하지 않음) — "방향이 맞는지" 수준의 신호이고 엄밀한 평가가 아니다. |
| GPU 메모리 로깅 | `torch.cuda.max_memory_allocated()`는 CUDA에서만 의미 있다 — CPU(스모크)에서는 항상 `N/A`. |
| 실제 RoCo 데이터셋 스키마 자체 | `docs/configurable_ard.md`가 확인한 블록 구조(44/14차원, state_select 등)에 근거해 설계했지만, **이 샌드박스는 그 데이터셋을 한 번도 직접 열어본 적이 없다** — 처음 실행에서 `validate_schema`가 던지는 에러 메시지를 사람이 반드시 읽고 판단할 것(통과했다고 안심하지 말 것 — 휴리스틱 통과가 "완전히 맞다"는 보장은 아니다). |

## 3. Colab에서 처음 돌릴 때 예상되는 실패 지점 (우선순위 순)

1. **`--dataset-repo-id`가 비공개/게이트 데이터셋인데 `HF_TOKEN`을 안 줬을 때** —
   `LeRobotDatasetMetadata`/`snapshot_download`가 403/401로 실패. Secrets에
   `HF_TOKEN`을 제대로 등록했는지 먼저 확인.
2. **`validate_schema`가 중단시킬 때** — `schema_check.py`의 휴리스미스틱이 실제
   RoCo 채널 이름과 안 맞아서(예: "left"/"right" 표기가 전혀 다른 컨벤션) 정상
   데이터인데도 중단되는 경우. 에러 메시지가 실제 채널 이름을 그대로 보여주니,
   `schema_check.py`의 정규식을 그 이름에 맞게 조정해야 할 수 있다.
3. **OOM(메모리 부족)** — `--batch-size auto` 추정치가 실측 없이 정한 값이라 T4
   16GB에서 넘칠 가능성이 있다. `--batch-size 1`로 낮춰서 재시도.
4. **`libsvtav1` 인코더/디코더 관련 에러** — `ffmpeg`에 해당 코덱이 없으면
   `LeRobotDataset`이 영상 디코딩에서 실패할 수 있다(미확인 영역). 에러가 나면
   `!apt-get install -y ffmpeg` 등으로 최신 ffmpeg를 깔아보거나, lerobot 쪽
   `video_backend` 옵션을 확인할 것.
5. **fp16 학습이 몇 스텝 뒤 loss가 NaN이 될 때** — T4에서 fp16 autocast+GradScaler를
   썼는데도 불안정하면, `--dtype fp32`로 강제하고(느리지만 안전) 원인을 좁혀갈 것.
6. **Drive 쓰기 권한/할당량 에러** — 체크포인트가 쌓이면서(기본 3개까지만 보관,
   `--keep-last-n-checkpoints`) Drive 용량을 압박할 수 있다. 데이터셋 자체도
   커서(`--drive-root/hf_cache`) Drive 무료 15GB를 넘기면 다운로드 중간에 실패한다
   — `fetch_dataset()`이 미리 용량을 로그로 보여주니 실행 전에 확인할 것.
