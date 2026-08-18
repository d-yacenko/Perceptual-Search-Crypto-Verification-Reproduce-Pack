#!/usr/bin/env python3
"""Reproduce the article's qualitative Gardens Point day/night figure.

The default command regenerates ``reports/real_visual_example.png`` and a
machine-readable sidecar with all 16 component decisions:

    python -B scripts/make_real_visual_example.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/article18_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from perceptual_crypto import keyed_fe_gen, keyed_verify_with_helper
from place_data import load_records
from run_place_keyed_component_experiment import build_components, component_bits
from dinov2_encoder import DinoV2TokenEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the article's Image001 Gardens Point component overlay."
    )
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "GardensPointWalking")
    parser.add_argument("--database-sequence", default="day_right")
    parser.add_argument("--query-sequence", default="night_right")
    parser.add_argument("--database-image", default="Image001.jpg")
    parser.add_argument("--query-image", default="Image001.jpg")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "real_visual_example.png")
    parser.add_argument("--metadata-output", type=Path, default=ROOT / "reports" / "real_visual_example.json")
    parser.add_argument("--model-name", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--device", default="cpu", help="PyTorch device; CPU is the canonical reproducibility mode.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--component-bits", type=int, default=4096)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--block-width", type=int, default=41)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max-errors-per-block", type=int, default=16)
    parser.add_argument("--max-total-errors", type=int, default=1000)
    return parser.parse_args()


def find_by_name(records, name: str) -> int:
    for i, r in enumerate(records):
        if Path(r.path).name == name:
            return i
    raise RuntimeError(f"{name} not found among {len(records)} records")


def main() -> int:
    cli = parse_args()
    dataset = cli.dataset_root.resolve()
    master_secret = "article18-visual-example-master-secret!!"

    args = SimpleNamespace(
        dataset_kind="gardens_point",
        dataset_root=dataset,
        download_gardens_point=False,
        gp_database_sequence=cli.database_sequence,
        gp_query_sequence=cli.query_sequence,
        max_database=0,
        max_queries=0,
        seed=42,
        sample_seed=0,
        manifest=None,
        split="test",
        positive_radius_m=25.0,
        negative_radius_m=100.0,
        sequence_positive_window=2,
    )

    print("Loading Gardens Point records...")
    db_records, q_records = load_records(args)
    day_i = find_by_name(db_records, cli.database_image)
    night_i = find_by_name(q_records, cli.query_image)
    day_path = Path(db_records[day_i].path)
    night_path = Path(q_records[night_i].path)
    print(
        f"pair: database[{day_i}]={day_path.name}  "
        f"query[{night_i}]={night_path.name}"
    )

    print(f"Encoding DINOv2 regional {cli.grid}x{cli.grid} components...")
    encoder = DinoV2TokenEncoder(
        model_name=cli.model_name,
        device=cli.device,
        image_size=cli.image_size,
    )
    day_cls, day_tokens = encoder.encode_paths([day_path], batch_size=1)
    night_cls, night_tokens = encoder.encode_paths([night_path], batch_size=1)

    day_comp = build_components(day_cls, day_tokens, "regional", cli.grid)
    night_comp = build_components(night_cls, night_tokens, "regional", cli.grid)
    day_bits = component_bits(day_comp, n_bits=cli.component_bits, seed=cli.projection_seed)[0]
    night_bits = component_bits(night_comp, n_bits=cli.component_bits, seed=cli.projection_seed)[0]

    real_matches: list[int] = []
    component_results: list[dict] = []
    for comp_idx in range(cli.grid * cli.grid):
        salt = f"gardens_point|{cli.database_sequence}|{day_path.name}|component:{comp_idx}"
        meta = f"record:{day_path.name}|component:{comp_idx}"
        enrollment = keyed_fe_gen(
            day_bits[comp_idx],
            master_secret=master_secret,
            salt=salt,
            meta=meta,
            blocks=cli.blocks,
            block_width=cli.block_width,
            overlap=cli.overlap,
            index_seed=2026 + comp_idx,
            max_errors_per_block=cli.max_errors_per_block,
            max_total_errors=cli.max_total_errors,
            secret_geometry=True,
        )
        result = keyed_verify_with_helper(
            night_bits[comp_idx],
            enrollment["helper"],
            enrollment["tag"],
            meta,
            master_secret=master_secret,
        )
        verified = int(result["verified"])
        real_matches.append(verified)
        component_results.append(
            {
                "database_component": comp_idx,
                "query_component": comp_idx,
                "verified": bool(result["verified"]),
                "decode_ok": bool(result["decode_ok"]),
                "tag_ok": bool(result["tag_ok"]),
                "corrected_observations": int(result["corrected_observations"]),
                "max_corrected_in_block": int(result["max_corrected_in_block"]),
                "overfull_blocks": int(result["overfull_blocks"]),
            }
        )

    print("real_matches =", real_matches)
    print(f"matched {sum(real_matches)}/{len(real_matches)}")

    img_day = Image.open(day_path)
    img_night = Image.open(night_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img_day)
    ax1.set_title("Эталонное место (База / День)", fontsize=16)
    ax1.axis("off")

    ax2.imshow(img_night)
    component_count = cli.grid * cli.grid
    ax2.set_title(f"Запрос (Ночь) — Совпало: {sum(real_matches)}/{component_count}", fontsize=16)
    ax2.axis("off")

    for ax_obj, img in ((ax1, img_day), (ax2, img_night)):
        w, h = img.size
        for i in range(1, cli.grid):
            ax_obj.axhline(i * h / cli.grid, color="white", linewidth=2, alpha=0.8)
            ax_obj.axvline(i * w / cli.grid, color="white", linewidth=2, alpha=0.8)

    w, h = img_night.size
    for i in range(cli.grid):
        for j in range(cli.grid):
            idx = i * cli.grid + j
            color = "#00ff00" if real_matches[idx] else "#ff0000"
            rect = patches.Rectangle(
                (j * w / cli.grid, i * h / cli.grid),
                w / cli.grid,
                h / cli.grid,
                linewidth=4,
                edgecolor=color,
                facecolor="none",
                alpha=0.9,
            )
            ax2.add_patch(rect)

    plt.tight_layout()
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cli.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    figure_sha256 = hashlib.sha256(cli.output.read_bytes()).hexdigest()
    metadata = {
        "figure": cli.output.name,
        "figure_sha256": figure_sha256,
        "dataset": "Gardens Point Walking",
        "database_image": str(day_path.relative_to(dataset)),
        "query_image": str(night_path.relative_to(dataset)),
        "matching_rule": "fixed_same_grid_cell_for_qualitative_overlay",
        "model_name": cli.model_name,
        "image_size": cli.image_size,
        "grid": cli.grid,
        "components": component_count,
        "component_bits": cli.component_bits,
        "projection_seed": cli.projection_seed,
        "secret_geometry": True,
        "blocks": cli.blocks,
        "block_width": cli.block_width,
        "overlap": cli.overlap,
        "max_errors_per_block": cli.max_errors_per_block,
        "max_total_errors": cli.max_total_errors,
        "verified_count": int(sum(real_matches)),
        "matches": real_matches,
        "component_results": component_results,
    }
    cli.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    cli.metadata_output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("saved", cli.output)
    print("saved", cli.metadata_output)
    print("figure_sha256 =", figure_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
