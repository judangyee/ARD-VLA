# 비전 토큰 프루닝 (`--use-token-pruning`, EfficientVLA식)

[← README로 돌아가기](../README.md)

SmolVLA는 프레임 하나를 SigLIP 인코더 + pixel shuffle(`transformers`의
`SmolVLMConnector.pixel_shuffle`, `scale_factor=2`) + modality projection을 거쳐 고정 개수의
비전 토큰(보통 64개)으로 만듭니다 (`SmolVLMWithExpertModel.embed_image()`,
`smolvlm_with_expert.py`). `lerobot/policies/smolvla/token_pruning.py`(신규)가 이 토큰
집합을 Task-Relevance and Diversity-Driven 방식(EfficientVLA, Yang et al. 2025의 설명을
바탕으로 직접 구현 — 논문 원문을 옮긴 게 아닙니다)으로 줄입니다:

1. 언어 지시문 토큰을 쿼리, 비전 토큰을 키로 한 (파라미터 없는) scaled dot-product
   cross-attention으로 토큰별 "태스크 관련성 점수"를 매깁니다 (`compute_task_relevance_scores`).
2. 관련성 상위 `token_pruning_k_key`개(기본 6, 논문 권장 4~8)는 핵심 세트로 무조건 남깁니다.
3. 남은 예산의 절반은 관련성 순위대로, 절반은 이미 뽑힌 토큰과 코사인 거리가 최대인(=가장 다른)
   토큰을 그리디하게 골라 채웁니다 — 다양성 확보 (`select_tokens`).

`SmolVLAConfig(use_token_pruning=True, token_pruning_k_final=32, token_pruning_k_key=6)`처럼
켜면 `VLAFlowMatching.embed_prefix()`가 이미지별로 자동 적용합니다 (학습/추론 양쪽 다 이
메서드 하나를 거치므로 별도 처리가 필요 없습니다). `token_pruning_k_final`이 실험 변수입니다 —
64에서 얼마나 더 줄일지 자유롭게 바꿔볼 수 있습니다.

`tests/test_token_pruning.py`(오프라인, 13개 통과 — 핵심 세트 보장, 다양성 채우기가 실제로
코사인 거리 최대 토큰을 고르는지, gather 기반이라 선택된 토큰에만 정확히 gradient가 흐르는지
등)에 더해, 실제 코드 경로(`SmolVLAPolicy.forward()` → `embed_prefix()`)를 소형 합성
SmolVLM 백본(Hub 접근 없이 `AutoConfig`/`AutoProcessor.from_pretrained`만 몽키패치, vision
`image_size=128`·`patch_size=8`로 pixel shuffle 후 정확히 64토큰/프레임이 되도록 맞춤)으로
K_final=32/48/64 각각 5스텝 돌려 확인했습니다 — 전부 정상 종료, loss 전부 유한.

한 가지 중요한 발견: 이 소형 합성 백본에서는 64개 토큰의 relevance score가 사실상 균일했습니다
(표준편차가 평균 대비 0.06% 수준) — 프루닝 로직의 결함이 아니라, 무작위 초기화된 백본은
비전-언어 임베딩 사이에 학습된 의미적 정렬이 전혀 없어서 고차원 랜덤 벡터의 내적이 다 비슷하게
나오기 때문입니다. 이 실험은 **코드 경로(선택 로직·gradient 흐름·K_final 스윕)가 정상
작동한다**는 것만 증명하며, "실제로 태스크 관련 토큰을 골라내는지"는 사전학습된 진짜 SmolVLM2
가중치로만 확인할 수 있습니다 — GPU/Hub 접근이 생기면 재확인이 필요합니다.
