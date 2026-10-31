"""`scripts/colab/run_colab_training.py --smoke` 통합 테스트 — 실제 `SmolVLAPolicy`
(작은 합성 백본)로 전 구간(GPU 감지/스키마 검증/정규화 통계/학습/체크포인트 재시작/
오프라인 평가)이 깨지지 않는지 확인한다. Hugging Face Hub/Google Drive 접근이 전혀
없다(합성 데이터만 사용) — 실제 RoCo 데이터셋/가중치로의 end-to-end 실행은 이 샌드박스
에서 검증할 수 없다(미확인, docs/colab_free_tier.md 참고)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "colab"))

import run_colab_training  # noqa: E402


def test_smoke_mode_passes_all_stages_and_produces_eval_report(tmp_path):
    exit_code = run_colab_training.main(
        ["--smoke", "--smoke-steps", "12", "--output-dir", str(tmp_path)]
    )
    assert exit_code == 0

    checkpoints = sorted((tmp_path / "checkpoints").glob("ckpt_step_*.pt"))
    assert len(checkpoints) > 0

    report_path = tmp_path / "eval_report.json"
    assert report_path.exists()
    import json

    report = json.loads(report_path.read_text())
    assert report["num_samples"] == 3
    assert "overall" in report["model"]
    assert "overall" in report["constant_mean_baseline"]
    assert "overall" in report["repeat_last_baseline"]


def test_smoke_mode_reports_failure_exit_code_on_broken_stage(tmp_path, monkeypatch):
    """스모크의 한 단계가 실제로 깨졌을 때 exit code가 0이 아니어야 한다(거짓 PASS 방지)."""
    import schema_check

    def _broken_validate_schema(info):
        raise schema_check.SchemaMismatchError("강제로 깨뜨린 테스트")

    monkeypatch.setattr(schema_check, "validate_schema", _broken_validate_schema)
    exit_code = run_colab_training.main(["--smoke", "--smoke-steps", "8", "--output-dir", str(tmp_path)])
    assert exit_code == 1
