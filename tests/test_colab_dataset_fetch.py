"""scripts/colab/dataset_fetch.py — 용량 조회/디스크 비교 로직. 실제 Hugging Face Hub
호출(`_dataset_info`/`_snapshot_download`)은 이 샌드박스에서 네트워크가 막혀 있어
실행할 수 없다(미검증) — 그 두 지점만 몽키패치해서 그 위의 로직(합산/비교/경고)을
검증한다."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import dataset_fetch  # noqa: E402


class _FakeSibling:
    def __init__(self, size):
        self.size = size


class _FakeDatasetInfo:
    def __init__(self, sizes):
        self.siblings = [_FakeSibling(s) for s in sizes]


def test_query_dataset_size_sums_known_sizes(monkeypatch):
    monkeypatch.setattr(dataset_fetch, "_dataset_info", lambda repo_id, token=None: _FakeDatasetInfo([100, 200, 300]))
    info = dataset_fetch.query_dataset_size("fake/repo")
    assert info.total_bytes == 600
    assert info.num_files == 3
    assert info.files_with_unknown_size == 0


def test_query_dataset_size_counts_unknown_sizes(monkeypatch):
    monkeypatch.setattr(dataset_fetch, "_dataset_info", lambda repo_id, token=None: _FakeDatasetInfo([100, None, 300]))
    info = dataset_fetch.query_dataset_size("fake/repo")
    assert info.total_bytes == 400
    assert info.files_with_unknown_size == 1


def test_compare_with_disk_fits_when_plenty_of_space(tmp_path):
    cmp = dataset_fetch.compare_with_disk(required_bytes=1024, check_path=tmp_path, safety_margin_gb=0.0)
    assert cmp.fits is True
    assert cmp.margin_gb > 0


def test_compare_with_disk_does_not_fit_when_required_exceeds_free(tmp_path):
    huge = 10**18  # 1 exabyte, 실제 디스크보다 항상 큼
    cmp = dataset_fetch.compare_with_disk(required_bytes=huge, check_path=tmp_path, safety_margin_gb=0.0)
    assert cmp.fits is False
    assert cmp.margin_gb < 0


def test_compare_with_disk_respects_safety_margin(tmp_path):
    free_gb = dataset_fetch.compare_with_disk(0, tmp_path, safety_margin_gb=0.0).free_gb
    # 안전 마진을 free 공간보다 크게 주면 0바이트 요청도 "안 맞음"이 돼야 한다.
    cmp = dataset_fetch.compare_with_disk(required_bytes=0, check_path=tmp_path, safety_margin_gb=free_gb + 1.0)
    assert cmp.fits is False


def test_fetch_dataset_calls_snapshot_download_with_cache_dir(tmp_path, monkeypatch):
    calls = {}

    monkeypatch.setattr(dataset_fetch, "_dataset_info", lambda repo_id, token=None: _FakeDatasetInfo([1024]))

    def fake_snapshot_download(repo_id, cache_dir, token=None, revision=None):
        calls["repo_id"] = repo_id
        calls["cache_dir"] = str(cache_dir)
        calls["token"] = token
        return str(Path(cache_dir) / "snapshot")

    monkeypatch.setattr(dataset_fetch, "_snapshot_download", fake_snapshot_download)

    cache_dir = tmp_path / "drive_cache"
    result = dataset_fetch.fetch_dataset("fake/repo", cache_dir, token="secret")

    assert calls["repo_id"] == "fake/repo"
    assert calls["cache_dir"] == str(cache_dir)
    assert calls["token"] == "secret"
    assert result == Path(cache_dir / "snapshot")
    assert cache_dir.exists()


def test_fetch_dataset_proceeds_even_when_disk_does_not_fit(tmp_path, monkeypatch):
    """용량이 부족해도 예외를 던지지 않고 경고만 남기고 계속 진행해야 한다(설계 의도 —
    모듈 docstring 참고: 더 엄격하게 막을지는 호출부의 선택)."""
    monkeypatch.setattr(dataset_fetch, "_dataset_info", lambda repo_id, token=None: _FakeDatasetInfo([10**18]))
    monkeypatch.setattr(dataset_fetch, "_snapshot_download", lambda repo_id, cache_dir, token=None, revision=None: str(cache_dir))

    result = dataset_fetch.fetch_dataset("fake/repo", tmp_path / "cache")
    assert result == tmp_path / "cache"
