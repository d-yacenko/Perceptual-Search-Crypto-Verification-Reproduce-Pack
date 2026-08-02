#!/usr/bin/env python3
"""Verify that the article's experimental claims match kept CSV artifacts."""

from __future__ import annotations

import base64
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perceptual_crypto import keyed_fe_gen, keyed_verify_with_helper

REPORTS = ROOT / "reports"
ARTICLE_CANDIDATES = [
    ROOT / "article18_protected_visual_matching_draft_ru.md",
    ROOT / "latex" / "article18_protected_visual_matching_ru.tex",
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise AssertionError(f"Missing report: {path.relative_to(ROOT)}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def threshold_row(prefix: str, threshold: int) -> Dict[str, str]:
    path = REPORTS / f"{prefix}_threshold_sweep.csv"
    for row in read_csv_rows(path):
        if int(row["min_verified_components"]) == threshold:
            return row
    raise AssertionError(f"Missing threshold M >= {threshold} in {path.relative_to(ROOT)}")


def pair_row(prefix: str, pair_set: str) -> Dict[str, str]:
    path = REPORTS / f"{prefix}_pair_summary.csv"
    for row in read_csv_rows(path):
        if row["pair_set"] == pair_set:
            return row
    raise AssertionError(f"Missing pair_set={pair_set!r} in {path.relative_to(ROOT)}")


def leakage(prefix: str) -> Dict[str, float]:
    path = REPORTS / f"{prefix}_leakage_summary.csv"
    return {row["metric"]: float(row["value"]) for row in read_csv_rows(path)}


def check_close(label: str, actual: float, expected: float, tol: float = 1e-6) -> int:
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")
    return 1


def check_equal(label: str, actual, expected) -> int:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")
    return 1


def read_article_text_for_checks() -> str:
    chunks = []
    for path in ARTICLE_CANDIDATES:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            normalized = (
                text
                .replace("\\_", "_")
                .replace("\\ ", " ")
                .replace("-\\/-", "--")
            )
            chunks.extend([text, normalized])
    if not chunks:
        candidates = ", ".join(str(p.relative_to(ROOT)) for p in ARTICLE_CANDIDATES)
        raise AssertionError(f"Missing article source, expected one of: {candidates}")
    return "\n".join(chunks)


def check_article_literals(literals: Iterable[object]) -> int:
    text = read_article_text_for_checks()
    count = 0
    for literal in literals:
        variants = literal if isinstance(literal, tuple) else (literal,)
        if not any(str(variant) in text for variant in variants):
            raise AssertionError(f"Article text does not contain expected literal: {variants[0]}")
        count += 1
    for stale in ["`8.762`", "`0.492`"]:
        if stale in text:
            raise AssertionError(f"Article still contains stale literal: {stale}")
    return count


def check_core_secret_geometry() -> int:
    rng = np.random.default_rng(20260501)
    code = rng.integers(0, 2, size=4096, dtype=np.uint8)
    master_secret = "article18-claim-check-secret"
    salt = base64.b64encode(b"article18-claim-check-salt").decode("ascii")
    meta = "claim-check-record"

    public = keyed_fe_gen(
        code,
        master_secret=master_secret,
        salt=salt,
        meta=meta,
        blocks=64,
        block_width=41,
        max_errors_per_block=16,
        max_total_errors=1000,
        index_seed=2026,
        secret_geometry=False,
    )
    secret = keyed_fe_gen(
        code,
        master_secret=master_secret,
        salt=salt,
        meta=meta,
        blocks=64,
        block_width=41,
        max_errors_per_block=16,
        max_total_errors=1000,
        index_seed=2026,
        secret_geometry=True,
    )

    count = 0
    count += check_equal("public helper exposes index_seed", "index_seed" in public["helper"], True)
    count += check_equal("secret helper hides index_seed", "index_seed" in secret["helper"], False)
    count += check_equal("public helper verifies source code", keyed_verify_with_helper(
        code,
        public["helper"],
        public["tag"],
        meta,
        master_secret=master_secret,
    )["verified"], True)
    count += check_equal("secret helper verifies source code", keyed_verify_with_helper(
        code,
        secret["helper"],
        secret["tag"],
        meta,
        master_secret=master_secret,
    )["verified"], True)
    return count


def main() -> int:
    dino_reg_sg0 = "protected_gardens_point_day_right_night_right_dinov2_vits14_regionalg4_bits4096_b64_w41_o0_mb16_mt1000_sg0"
    dino_reg_sg1 = "protected_gardens_point_day_right_night_right_dinov2_vits14_regionalg4_bits4096_b64_w41_o0_mb16_mt1000_sg1"
    dino_cls_sg1 = "protected_gardens_point_day_right_night_right_dinov2_vits14_cls_bits4096_b64_w41_o0_mb16_mt1000_sg1"
    vit_sg1 = "protected_gardens_point_day_right_night_right_vit_b_16_regionalg4_bits4096_b64_w41_o0_mb16_mt1000_sg1"
    resnet_sg1 = "protected_gardens_point_day_right_night_right_resnet50_regionalg4_bits4096_b64_w41_o0_mb16_mt1000_sg1"
    densenet_sg1 = "protected_gardens_point_day_right_night_right_densenet121_regionalg4_bits4096_b64_w41_o0_mb16_mt1000_sg1"

    checked = 0

    expected_thresholds = {
        dino_reg_sg0: {
            4: {"tpr": 0.8175, "far": 0.0400, "helper_only_tpr": 0.8175, "helper_only_far": 0.0400},
            6: {"tpr": 0.7125, "far": 0.0125, "helper_only_tpr": 0.7125, "helper_only_far": 0.0125},
            8: {"tpr": 0.5900, "far": 0.0050, "helper_only_tpr": 0.5900, "helper_only_far": 0.0050},
        },
        dino_reg_sg1: {
            4: {"tpr": 0.8300, "far": 0.0350, "frr": 0.1700, "helper_only_tpr": 0.0, "helper_only_far": 0.0},
            6: {"tpr": 0.7075, "far": 0.0125, "frr": 0.2925, "helper_only_tpr": 0.0, "helper_only_far": 0.0},
            8: {"tpr": 0.5950, "far": 0.0025, "frr": 0.4050, "helper_only_tpr": 0.0, "helper_only_far": 0.0},
        },
        dino_cls_sg1: {
            1: {"tpr": 0.2000, "far": 0.0025, "frr": 0.8000},
        },
        vit_sg1: {
            8: {"tpr": 0.5050, "far": 0.0200},
        },
        densenet_sg1: {
            4: {"tpr": 0.3800, "far": 0.0025},
        },
        resnet_sg1: {
            3: {"tpr": 0.0600, "far": 0.0025},
        },
    }

    for prefix, by_threshold in expected_thresholds.items():
        for threshold, expected_values in by_threshold.items():
            row = threshold_row(prefix, threshold)
            for metric, expected in expected_values.items():
                checked += check_close(f"{prefix} M>={threshold} {metric}", float(row[metric]), expected)

    expected_leakage = {
        dino_reg_sg0: {
            "helper_only_component_count_auc": 0.948994,
            "helper_only_best_penalty_auc": 0.955175,
        },
        dino_reg_sg1: {
            "helper_only_component_count_auc": 0.500000,
            "helper_only_best_penalty_auc": 0.500000,
            "verified_component_count_auc": 0.951856,
        },
        dino_cls_sg1: {
            "helper_only_component_count_auc": 0.500000,
        },
        vit_sg1: {
            "helper_only_component_count_auc": 0.500000,
            "verified_component_count_auc": 0.890891,
        },
        densenet_sg1: {
            "helper_only_component_count_auc": 0.500000,
            "verified_component_count_auc": 0.819913,
        },
        resnet_sg1: {
            "helper_only_component_count_auc": 0.500000,
            "verified_component_count_auc": 0.613234,
        },
    }

    for prefix, expected_values in expected_leakage.items():
        values = leakage(prefix)
        for metric, expected in expected_values.items():
            checked += check_close(f"{prefix} {metric}", values[metric], expected)

    expected_pairs = {
        (dino_reg_sg1, "positive_place"): {
            "n_pairs": 400,
            "components_per_image": 16,
            "match_count_mean": 8.7625,
            "match_count_q90": 15.0,
            "match_count_max": 16.0,
            "helper_only_count_mean": 0.0,
        },
        (dino_reg_sg1, "negative_place"): {
            "n_pairs": 400,
            "components_per_image": 16,
            "match_count_mean": 0.4925,
            "match_count_q90": 2.0,
            "match_count_max": 8.0,
            "helper_only_count_mean": 0.0,
        },
        (vit_sg1, "positive_place"): {"match_count_mean": 7.4525, "match_count_q90": 12.0},
        (vit_sg1, "negative_place"): {"match_count_mean": 2.3150, "match_count_q90": 5.0},
        (densenet_sg1, "positive_place"): {"match_count_mean": 2.9475, "match_count_q90": 7.0},
        (densenet_sg1, "negative_place"): {"match_count_mean": 0.1225, "match_count_q90": 0.0},
        (resnet_sg1, "positive_place"): {"match_count_mean": 0.5175, "match_count_q90": 2.0},
        (resnet_sg1, "negative_place"): {"match_count_mean": 0.0350, "match_count_q90": 0.0},
    }

    for (prefix, pair_set), expected_values in expected_pairs.items():
        row = pair_row(prefix, pair_set)
        for metric, expected in expected_values.items():
            checked += check_close(f"{prefix} {pair_set} {metric}", float(row[metric]), expected)

    # Check the article-facing comparative statements.
    checked += check_equal(
        "DINOv2 regional M>=8 TPR beats CLS M>=1 TPR",
        float(threshold_row(dino_reg_sg1, 8)["tpr"]) > float(threshold_row(dino_cls_sg1, 1)["tpr"]),
        True,
    )
    checked += check_equal(
        "ViT-B/16 M>=8 TPR beats DenseNet-121 M>=4 TPR",
        float(threshold_row(vit_sg1, 8)["tpr"]) > float(threshold_row(densenet_sg1, 4)["tpr"]),
        True,
    )
    checked += check_equal(
        "DenseNet-121 M>=4 TPR beats ResNet-50 M>=3 TPR at same FAR",
        float(threshold_row(densenet_sg1, 4)["tpr"]) > float(threshold_row(resnet_sg1, 3)["tpr"])
        and float(threshold_row(densenet_sg1, 4)["far"]) == float(threshold_row(resnet_sg1, 3)["far"]),
        True,
    )

    checked += check_article_literals(
        [
            "TPR = 0.8300",
            "FAR = 0.0350",
            "TPR = 0.7075",
            "FAR = 0.0125",
            "TPR = 0.5950",
            "FAR = 0.0025",
            "0.948994--0.955175",
            "0.500000",
            ("`8.7625`", "8.7625"),
            ("`0.4925`", "0.4925"),
            "scripts/run_place_keyed_component_experiment.py",
            "scripts/run_place_torchvision_component_experiment.py",
        ]
    )

    checked += check_core_secret_geometry()

    stale_reports = list(REPORTS.glob("*.md"))
    checked += check_equal("no stale report artifacts", [p.name for p in stale_reports], [])

    print(f"OK: verified {checked} article claims against CSV artifacts and Python implementation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
