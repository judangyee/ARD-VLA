"""에피소드를 학습/평가로 고정 seed로 나누고, 그 목록을 파일로 저장/복원한다.

재현성이 핵심이다 — 같은 `num_episodes`/`num_eval`/`seed`면 몇 번을 다시 실행해도
항상 같은 분할이 나와야 한다(세션이 끊겨도 다시 받은 데이터셋에서 "학습 190/평가 10"이
매번 똑같아야 체크포인트 재시작이 의미 있다)."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpisodeSplit:
    num_episodes: int
    num_eval: int
    seed: int
    train_ids: list[int]
    eval_ids: list[int]


def split_episodes(num_episodes: int, num_eval: int = 10, seed: int = 0) -> EpisodeSplit:
    if num_eval < 0 or num_eval > num_episodes:
        raise ValueError(f"num_eval({num_eval})은 0 이상, num_episodes({num_episodes}) 이하여야 합니다.")

    all_ids = list(range(num_episodes))
    rng = random.Random(seed)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)

    eval_ids = sorted(shuffled[:num_eval])
    train_ids = sorted(shuffled[num_eval:])
    return EpisodeSplit(
        num_episodes=num_episodes, num_eval=num_eval, seed=seed, train_ids=train_ids, eval_ids=eval_ids
    )


def save_split(path: str | Path, split: EpisodeSplit) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(split), ensure_ascii=False, indent=2))


def load_split(path: str | Path) -> EpisodeSplit:
    data = json.loads(Path(path).read_text())
    return EpisodeSplit(**data)


def load_or_create_split(path: str | Path, num_episodes: int, num_eval: int = 10, seed: int = 0) -> EpisodeSplit:
    """`path`에 분할 파일이 이미 있으면 그걸 그대로 쓴다(재시작 시 데이터셋을 다시 받아도
    같은 분할을 유지하기 위함 — `num_episodes`가 저장된 값과 다르면 데이터셋이 바뀐
    것이므로 명확한 에러로 중단한다). 없으면 새로 만들어서 저장한다."""
    path = Path(path)
    if path.exists():
        existing = load_split(path)
        if existing.num_episodes != num_episodes:
            raise ValueError(
                f"저장된 분할 파일({path})의 num_episodes={existing.num_episodes}이 현재 데이터셋의 "
                f"에피소드 수({num_episodes})와 다릅니다 — 데이터셋이 바뀌었을 수 있습니다. "
                "분할 파일을 지우고 다시 만들거나, 올바른 파일인지 확인하세요."
            )
        return existing

    split = split_episodes(num_episodes, num_eval=num_eval, seed=seed)
    save_split(path, split)
    return split
