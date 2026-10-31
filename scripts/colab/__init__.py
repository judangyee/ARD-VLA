"""무료 Colab에서 RoCo 데이터로 스모크/베이스라인 학습을 돌리기 위한 보조 모듈 묶음.

각 모듈은 독립적으로 단위 테스트가 가능하도록 순수 함수 위주로 짰다(실제 GPU/Hugging
Face Hub/Google Drive 접근 없이 `tests/test_colab_*.py`에서 검증). 실행 진입점은
`scripts/colab/run_colab_training.py` 하나다 — 자세한 사용법은
`docs/colab_free_tier.md` 참고.
"""
