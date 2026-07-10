from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CleanSample:
    staging_path: Path
    identity: str
    original_path: str
    relative_output_path: str
    metrics: dict[str, float]
    augmentation_regions: dict[str, list[int]]


def identity_level_split(
    samples: list[CleanSample],
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 23,
) -> dict[str, list[CleanSample]]:
    if not samples:
        return {"train": [], "validation": [], "test": []}

    total_ratio = train_ratio + validation_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    train_ratio /= total_ratio
    validation_ratio /= total_ratio

    by_identity: dict[str, list[CleanSample]] = defaultdict(list)
    for sample in samples:
        by_identity[sample.identity].append(sample)

    identities = sorted(by_identity)
    rng = random.Random(seed)
    rng.shuffle(identities)

    n = len(identities)
    if n == 1:
        split_ids = {"train": identities, "validation": [], "test": []}
    else:
        train_count = max(1, int(round(n * train_ratio)))
        validation_count = int(round(n * validation_ratio))
        if train_count + validation_count >= n:
            validation_count = max(0, n - train_count - 1)
        split_ids = {
            "train": identities[:train_count],
            "validation": identities[train_count:train_count + validation_count],
            "test": identities[train_count + validation_count:],
        }

    return {
        split: [sample for identity in split_identities for sample in by_identity[identity]]
        for split, split_identities in split_ids.items()
    }


def class_distribution(samples: list[CleanSample]) -> dict[str, int]:
    return dict(sorted(Counter(sample.identity for sample in samples).items()))


def split_summary(splits: dict[str, list[CleanSample]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            "images": len(split_samples),
            "identities": len({sample.identity for sample in split_samples}),
        }
        for split, split_samples in splits.items()
    }

