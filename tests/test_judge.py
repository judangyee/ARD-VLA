"""ARD-VLA 폐루프 구조의 Judge/Selector head(`lerobot.policies.smolvla.judge.Judge`)에 대한
오프라인 단위 테스트 — `tests/test_ard.py`와 같은 스타일로 합성 텐서를 직접 넣어 검증한다.
실제 SmolVLAPolicy 전체(prefix forward 1회 공유, 학습 손실 연결)를 검증하는 통합 테스트는
`tests/test_ard_integration.py`에 있다.
"""

from __future__ import annotations

import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.judge import Judge


def _make_prefix(batch=3, seq_len=10, valid_len=6, hidden=16):
    hidden_state = torch.randn(batch, seq_len, hidden)
    pad_masks = torch.zeros(batch, seq_len, dtype=torch.bool)
    pad_masks[:, :valid_len] = True
    return hidden_state, pad_masks


def test_judge_only_builds_requested_heads():
    judge = Judge(vlm_hidden_size=16, use_judge=True, num_failure_types=0, use_progress=False, use_selector=False)
    assert judge.success_head is not None
    assert judge.failure_head is None
    assert judge.progress_head is None
    assert judge.selector_head is None

    selector_only = Judge(vlm_hidden_size=16, use_judge=False, use_selector=True, selector_num_options=4)
    assert selector_only.success_head is None
    assert selector_only.selector_head is not None


def test_judge_output_shapes_and_ranges():
    hidden_state, pad_masks = _make_prefix()
    judge = Judge(
        vlm_hidden_size=16, use_judge=True, num_failure_types=3, use_progress=True,
        use_selector=True, selector_num_options=5,
    )
    out = judge(hidden_state, pad_masks)

    assert out["success_prob"].shape == (3,)
    assert torch.all((out["success_prob"] >= 0) & (out["success_prob"] <= 1))

    assert out["failure_probs"].shape == (3, 3)
    assert torch.allclose(out["failure_probs"].sum(dim=-1), torch.ones(3), atol=1e-5)

    assert out["progress"].shape == (3,)
    assert torch.all((out["progress"] >= 0) & (out["progress"] <= 1))

    assert out["selector_probs"].shape == (3, 5)
    assert torch.allclose(out["selector_probs"].sum(dim=-1), torch.ones(3), atol=1e-5)


def test_judge_masked_mean_pooling_ignores_padding():
    """패딩 구간(valid_len 뒤)의 값을 바꿔도 pooling 결과(=success_prob)가 변하면 안 된다."""
    hidden_state, pad_masks = _make_prefix(batch=2, seq_len=10, valid_len=4, hidden=16)
    judge = Judge(vlm_hidden_size=16, use_judge=True)
    judge.eval()

    out_before = judge(hidden_state, pad_masks)
    hidden_state_perturbed = hidden_state.clone()
    hidden_state_perturbed[:, 4:, :] = 999.0  # 패딩 구간만 바꿈
    out_after = judge(hidden_state_perturbed, pad_masks)

    assert torch.allclose(out_before["success_prob"], out_after["success_prob"], atol=1e-6)


def test_judge_temperature_scales_success_prob():
    hidden_state, pad_masks = _make_prefix(batch=1, seq_len=10, valid_len=10, hidden=16)
    judge = Judge(vlm_hidden_size=16, use_judge=True)
    judge.eval()

    with torch.no_grad():
        out_t1 = judge(hidden_state, pad_masks)
        logit = out_t1["success_logit"]
        judge.temperature.fill_(4.0)
        out_t4 = judge(hidden_state, pad_masks)

    expected = torch.sigmoid(logit / 4.0)
    assert torch.allclose(out_t4["success_prob"], expected, atol=1e-6)
    assert not torch.allclose(out_t1["success_prob"], out_t4["success_prob"], atol=1e-6)


def test_judge_selector_allowed_mask_zeroes_disallowed_options():
    hidden_state, pad_masks = _make_prefix(batch=2, seq_len=10, valid_len=10, hidden=16)
    judge = Judge(vlm_hidden_size=16, use_judge=False, use_selector=True, selector_num_options=4)

    allowed_mask = torch.tensor([True, False, True, False])
    out = judge(hidden_state, pad_masks, allowed_mask=allowed_mask)

    assert torch.allclose(out["selector_probs"][:, 1], torch.zeros(2), atol=1e-7)
    assert torch.allclose(out["selector_probs"][:, 3], torch.zeros(2), atol=1e-7)
    assert torch.allclose(out["selector_probs"].sum(dim=-1), torch.ones(2), atol=1e-5)


def test_judge_detach_features_blocks_gradient_to_input():
    hidden_state, pad_masks = _make_prefix(batch=2, seq_len=10, valid_len=10, hidden=16)
    hidden_state.requires_grad_(True)

    judge_detached = Judge(vlm_hidden_size=16, use_judge=True, detach_features=True)
    out = judge_detached(hidden_state, pad_masks)
    out["success_prob"].sum().backward()
    assert hidden_state.grad is None

    hidden_state2 = hidden_state.detach().clone().requires_grad_(True)
    judge_attached = Judge(vlm_hidden_size=16, use_judge=True, detach_features=False)
    out2 = judge_attached(hidden_state2, pad_masks)
    out2["success_prob"].sum().backward()
    assert hidden_state2.grad is not None


def _base_kwargs(**overrides):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE

    kwargs = dict(
        input_features={
            OBS_IMAGE: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
    )
    kwargs.update(overrides)
    return kwargs


def test_config_judge_defaults_off():
    config = SmolVLAConfig(**_base_kwargs())
    assert config.use_judge is False
    assert config.use_selector is False
    assert config.judge_feature_layer == -1


def test_config_selector_requires_num_options():
    with pytest.raises(ValueError, match="selector_num_options"):
        SmolVLAConfig(**_base_kwargs(use_selector=True, selector_num_options=0))


def test_config_selector_num_options_requires_use_selector():
    with pytest.raises(ValueError, match="use_selector"):
        SmolVLAConfig(**_base_kwargs(use_selector=False, selector_num_options=3))


def test_config_judge_failure_types_requires_use_judge():
    with pytest.raises(ValueError, match="use_judge"):
        SmolVLAConfig(**_base_kwargs(use_judge=False, judge_num_failure_types=2))


def test_config_judge_progress_requires_use_judge():
    with pytest.raises(ValueError, match="use_judge"):
        SmolVLAConfig(**_base_kwargs(use_judge=False, judge_use_progress=True))


def test_config_judge_feature_layer_rejects_negative_non_minus_one():
    with pytest.raises(ValueError, match="judge_feature_layer"):
        SmolVLAConfig(**_base_kwargs(judge_feature_layer=-2))
