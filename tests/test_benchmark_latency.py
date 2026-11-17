"""`scripts/benchmark_latency.py`의 지연/메모리/파라미터 측정 로직을 tiny 합성 백본으로
검증한다(CPU, 실제 Hub/GPU 없음). GPU 피크 메모리 측정 자체(`torch.cuda.*`)는 이 샌드박스에
GPU가 없어 검증 불가 — `peak_gpu_mem_gb`가 CPU에서 `None`으로 나오는지만 확인한다
(docs/colab_free_tier.md의 "GPU 메모리 로깅은 CUDA에서만 의미있다" 관례와 동일)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_latency  # noqa: E402


def _build_offline_tokenizer():
    """tests/test_policy_interface.py의 `_build_offline_tokenizer`와 같은 패턴(파일을
    가로질러 import하지 않고 각 테스트 파일이 자급자족하도록 각자 둔다)."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, "insert": 2, "the": 3, "bolt": 4, "into": 5, "board": 6}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")


def _patch_tiny_backbone(monkeypatch):
    """`AutoConfig`/`AutoProcessor`/`AutoTokenizer.from_pretrained`를 네트워크 없는 tiny
    SmolVLM 구성으로 몽키패치한다 — tests/test_policy_interface.py와 동일한 패턴."""
    import transformers
    from transformers import SmolVLMConfig, SmolVLMVisionConfig

    tiny_cfg = SmolVLMConfig(
        vision_config=SmolVLMVisionConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_channels=3, image_size=64, patch_size=16,
        ),
        text_config={
            "model_type": "llama", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": 256, "max_position_embeddings": 128,
        },
        scale_factor=2, image_token_id=57,
    )

    class _StubTok:
        fake_image_token_id = 1
        global_image_token_id = 2

    class _StubProc:
        tokenizer = _StubTok()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(lambda *a, **k: tiny_cfg))
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", staticmethod(lambda *a, **k: _StubProc()))
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: _build_offline_tokenizer())
    )


def _random_init_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        checkpoint_path=None,
        random_init=True,
        preset=str(benchmark_latency.ROCO_PRESET_PATH),
        vlm_model_name="fake/tiny-smolvlm",
        num_vlm_layers=2,
        image_height=64,
        image_width=64,
        task="insert the bolt into the board",
        denoise_steps=[1, 2],
        num_samples=2,
        num_warmup=1,
        device=None,
        seed=0,
        output_path="outputs/_test_benchmark_latency.json",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_policy_random_init_uses_roco_preset(monkeypatch):
    """--random-init이 실제 configs/presets/roco_board.yaml을 그대로 적용하는지(ard_arm_dim=7
    -> action_dim=14, chunk_size=20/n_action_steps=5)."""
    _patch_tiny_backbone(monkeypatch)
    args = _random_init_args()
    policy = benchmark_latency.build_policy(args, torch.device("cpu"))

    assert policy.config.chunk_size == 20
    assert policy.config.n_action_steps == 5
    assert policy.config.ard_arm_dim == 7
    assert policy.config.action_feature.shape[0] == benchmark_latency.ACTION_DIM


def test_count_params_reflects_frozen_vision_encoder(monkeypatch):
    """roco_board.yaml의 freeze_vision_encoder=true가 실제로 학습 가능 파라미터 수를
    전체보다 줄이는지 — count_params()가 그 차이를 그대로 보여주는지."""
    _patch_tiny_backbone(monkeypatch)
    args = _random_init_args()
    policy = benchmark_latency.build_policy(args, torch.device("cpu"))

    total, trainable = benchmark_latency.count_params(policy)
    assert total > 0
    assert 0 < trainable < total


def test_measure_latency_for_num_steps_produces_sane_stats(monkeypatch):
    """지연 측정 자체가 끝까지 돌고, median<=p95이며 peak_gpu_mem_gb가 CPU에서는 None인지."""
    import numpy as np

    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    from policy_interface import DEFAULT_CAMERA_KEYS, RoCoPolicyAdapter, STATE_DIM

    _patch_tiny_backbone(monkeypatch)
    args = _random_init_args()
    device = torch.device("cpu")
    policy = benchmark_latency.build_policy(args, device)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(policy.config, dataset_stats=None)
    adapter = RoCoPolicyAdapter(policy, preprocessor, postprocessor)

    rng = np.random.default_rng(0)
    images = {role: rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8) for role in DEFAULT_CAMERA_KEYS}
    state = rng.normal(size=(STATE_DIM,)).astype(np.float32)

    result = benchmark_latency.measure_latency_for_num_steps(
        adapter, images, state, "insert the bolt into the board",
        num_steps=2, num_warmup=1, num_samples=3, device=device,
    )

    assert result["num_denoise_steps"] == 2
    assert len(result["latency_ms_samples"]) == 3
    assert result["latency_ms_median"] <= result["latency_ms_p95"]
    assert all(ms > 0 for ms in result["latency_ms_samples"])
    assert result["peak_gpu_mem_gb"] is None
    assert policy.config.num_steps == 2  # 스윕이 실제로 config.num_steps를 바꿔놓고 끝남


def test_run_benchmark_sweeps_all_denoise_steps_and_matches_json_schema(monkeypatch, tmp_path):
    """run_benchmark()가 --denoise-steps 전부에 대해 한 건씩 결과를 내고, JSON으로 그대로
    직렬화 가능한지(별도의 커스텀 타입 없이 전부 기본 파이썬/float)."""
    import json

    _patch_tiny_backbone(monkeypatch)
    output_path = tmp_path / "bench.json"
    args = _random_init_args(denoise_steps=[1, 2, 5], output_path=str(output_path))

    report = benchmark_latency.run_benchmark(args)

    assert [r["num_denoise_steps"] for r in report["denoise_step_sweep"]] == [1, 2, 5]
    assert report["batch_size"] == 1
    assert report["total_params"] > 0
    assert report["trainable_params"] > 0
    # JSON으로 왕복 가능한지(main()이 하는 일과 동일한 직렬화 검증)
    json.loads(json.dumps(report))


def test_checkpoint_path_round_trip(monkeypatch, tmp_path):
    """save_pretrained()로 저장한 tiny 체크포인트를 --checkpoint-path로 다시 불러와도
    동일한 구조(action_dim/chunk_size)가 나오는지 — --random-init이 아닌 코드 경로 검증."""
    _patch_tiny_backbone(monkeypatch)
    random_init_args = _random_init_args()
    policy = benchmark_latency.build_policy(random_init_args, torch.device("cpu"))

    ckpt_dir = tmp_path / "ckpt"
    policy.save_pretrained(ckpt_dir)

    loaded_args = argparse.Namespace(checkpoint_path=str(ckpt_dir), random_init=False)
    loaded_policy = benchmark_latency.build_policy(loaded_args, torch.device("cpu"))

    assert loaded_policy.config.chunk_size == policy.config.chunk_size
    assert loaded_policy.config.action_feature.shape == policy.config.action_feature.shape


def test_main_requires_exactly_one_of_checkpoint_or_random_init(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["benchmark_latency.py"])
    try:
        benchmark_latency.main()
        raise AssertionError("둘 다 안 줬는데 에러가 안 났다")
    except SystemExit:
        pass

    monkeypatch.setattr(
        sys, "argv", ["benchmark_latency.py", "--random-init", "--checkpoint-path", "x"]
    )
    try:
        benchmark_latency.main()
        raise AssertionError("둘 다 줬는데 에러가 안 났다")
    except SystemExit:
        pass
