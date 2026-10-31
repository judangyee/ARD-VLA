"""RoCo 데이터셋(`rocochallenge2025/rocochallenge2026_Industrial_Assembly`)을 영상 포함해서
받는다 — Google Drive를 Hugging Face Hub 캐시 디렉터리로 써서, Colab 세션이 끊겨도
(로컬 디스크는 날아가지만 Drive는 남음) 다음 세션에서 다시 받을 필요가 없게 한다.

**이 모듈의 실제 네트워크 호출(`HfApi.dataset_info`/`snapshot_download`)은 이 샌드박스가
Hugging Face Hub 접근이 막혀 있어 한 번도 실행해보지 못했다 — 미검증.** 그래서 두 호출을
각각 작은 함수(`_dataset_info`/`_snapshot_download`)로 분리해서, 그 위의 로직(용량 계산,
디스크 여유 비교, 경고 문구)은 네트워크 없이도 몽키패치로 단위 테스트할 수 있게 했다
(`tests/test_colab_dataset_fetch.py`)."""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path


@dataclasses.dataclass
class DatasetSizeInfo:
    repo_id: str
    total_bytes: int
    num_files: int
    files_with_unknown_size: int


@dataclasses.dataclass
class DiskComparison:
    required_bytes: int
    free_bytes: int
    free_gb: float
    required_gb: float
    fits: bool
    margin_gb: float  # fits면 남는 여유, 아니면 부족분(음수)


def _dataset_info(repo_id: str, token: str | None = None):
    """실제 HfApi 호출을 감싼 지점 — 테스트에서 몽키패치 대상."""
    from huggingface_hub import HfApi

    return HfApi().dataset_info(repo_id, files_metadata=True, token=token)


def query_dataset_size(repo_id: str, token: str | None = None) -> DatasetSizeInfo:
    """repo의 전체 파일 용량을 합산한다. `files_metadata=True`로 각 파일의 `size`를
    받는데, 서버가 일부 파일의 size를 안 줄 수도 있어(LFS 메타데이터 없음 등) 그 개수도
    같이 보고한다 — 0이 아니면 `total_bytes`가 실제보다 작게 잡힐 수 있다는 뜻이므로
    로그에서 사람이 볼 수 있게 한다."""
    info = _dataset_info(repo_id, token=token)
    siblings = getattr(info, "siblings", None) or []
    total = 0
    unknown = 0
    for s in siblings:
        size = getattr(s, "size", None)
        if size is None:
            unknown += 1
        else:
            total += size
    return DatasetSizeInfo(
        repo_id=repo_id, total_bytes=total, num_files=len(siblings), files_with_unknown_size=unknown
    )


def compare_with_disk(required_bytes: int, check_path: str | Path, safety_margin_gb: float = 2.0) -> DiskComparison:
    """`check_path`가 있는 파티션의 여유 공간과 `required_bytes`를 비교한다.
    `safety_margin_gb`(기본 2GB)만큼은 항상 남겨둬야 "fits"로 판정한다 — 체크포인트/
    로그/임시 파일 쓸 공간이 필요하기 때문."""
    usage = shutil.disk_usage(Path(check_path).parent if not Path(check_path).exists() else check_path)
    free_gb = usage.free / (1024**3)
    required_gb = required_bytes / (1024**3)
    fits = free_gb - required_gb >= safety_margin_gb
    margin_gb = free_gb - required_gb - safety_margin_gb
    return DiskComparison(
        required_bytes=required_bytes,
        free_bytes=usage.free,
        free_gb=free_gb,
        required_gb=required_gb,
        fits=fits,
        margin_gb=margin_gb,
    )


def _snapshot_download(repo_id: str, cache_dir: str | Path, token: str | None = None, revision: str | None = None) -> str:
    """실제 다운로드를 감싼 지점 — 테스트에서 몽키패치 대상."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id, repo_type="dataset", cache_dir=str(cache_dir), token=token, revision=revision
    )


def fetch_dataset(
    repo_id: str,
    drive_cache_dir: str | Path,
    token: str | None = None,
    revision: str | None = None,
    min_free_gb_after: float = 2.0,
) -> Path:
    """용량을 먼저 조회해서 로그로 남기고, `drive_cache_dir`(Google Drive 위 경로를 넘기면
    세션이 끊겨도 보존됨)를 HF Hub 캐시로 써서 `snapshot_download`한다. `snapshot_download`
    자체가 콘텐츠 주소 기반 캐시라 이미 받은 파일은 재다운로드하지 않는다 — 그래서 Drive
    캐시를 재사용하는 것만으로 "세션이 끊겨도 이어받기"가 대부분 해결된다(완전한 재개가
    아니라, 이미 받은 파일을 다시 받지 않는다는 뜻 — `snapshot_download`가 파일 단위
    atomic 다운로드이므로, 받던 중 끊긴 파일 하나만 다시 받으면 된다).

    용량이 Drive 여유 공간보다 크면 예외를 던지지 않고(사용자가 Drive 용량을 늘리거나
    일부만 받기로 결정할 수 있어야 하므로) `DiskComparison.fits=False`를 로그로 경고만
    하고 계속 진행한다 — 호출자(`run_colab_training.py`)가 `--require-disk-fit`류
    플래그로 더 엄격하게 막을지는 호출부의 선택으로 남긴다.
    """
    import logging

    size_info = query_dataset_size(repo_id, token=token)
    logging.info(
        "데이터셋 %s 전체 용량: %.2f GB (%d개 파일, size 미확인 %d개)",
        repo_id, size_info.total_bytes / (1024**3), size_info.num_files, size_info.files_with_unknown_size,
    )

    Path(drive_cache_dir).mkdir(parents=True, exist_ok=True)
    disk_cmp = compare_with_disk(size_info.total_bytes, drive_cache_dir, safety_margin_gb=min_free_gb_after)
    if disk_cmp.fits:
        logging.info("디스크 여유 충분: 여유 %.2f GB, 필요 %.2f GB (여분 %.2f GB)", disk_cmp.free_gb, disk_cmp.required_gb, disk_cmp.margin_gb)
    else:
        logging.warning(
            "디스크 여유 부족 가능성: 여유 %.2f GB, 필요 %.2f GB (부족 %.2f GB) — 계속 진행하지만 "
            "다운로드 중 공간 부족 에러가 날 수 있습니다.",
            disk_cmp.free_gb, disk_cmp.required_gb, -disk_cmp.margin_gb,
        )

    path = _snapshot_download(repo_id, cache_dir=drive_cache_dir, token=token, revision=revision)
    return Path(path)
