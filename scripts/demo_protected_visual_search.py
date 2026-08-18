#!/usr/bin/env python3
"""Small end-to-end demo of protected visual place matching.

The script builds a small protected database from reference images, discards
open descriptors and binary visual codes, then scans the protected records for
several query images. It is meant as a readable prototype demo, not as the
article metric runner.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "article18_matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from perceptual_crypto import keyed_fe_gen, keyed_verify_with_helper
from place_data import PlaceRecord, load_records, relation_indices
from run_place_keyed_component_experiment import build_components, component_bits
from dinov2_encoder import DinoV2TokenEncoder


@dataclass
class ProtectedComponent:
    helper: dict
    tag: str
    meta: str


@dataclass
class ProtectedRecord:
    public_id: str
    components: List[ProtectedComponent]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small protected visual DB and scan it with query images."
    )
    parser.add_argument("--dataset-kind", choices=["gardens_point", "vpr_standard", "manifest"], default="gardens_point")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "GardensPointWalking")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--download-gardens-point", action="store_true")
    parser.add_argument("--gp-database-sequence", default="day_right")
    parser.add_argument("--gp-query-sequence", default="night_right")
    parser.add_argument("--database-size", type=int, default=12)
    parser.add_argument("--query-count", type=int, default=3)
    parser.add_argument("--database-start", type=int, default=0)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--selection-mode", choices=["spread", "contiguous"], default="spread")
    parser.add_argument("--model-name", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--device", default="cpu", help="PyTorch device; CPU is the canonical reproducibility mode.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--component-grid", type=int, default=4)
    parser.add_argument("--component-bits", type=int, default=4096)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--master-secret", default="article18-demo-master-secret-32-bytes")
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--block-width", type=int, default=41)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max-errors-per-block", type=int, default=16)
    parser.add_argument("--max-total-errors", type=int, default=1000)
    parser.add_argument("--min-components", type=int, default=6)
    parser.add_argument("--positive-radius-m", type=float, default=25.0)
    parser.add_argument("--negative-radius-m", type=float, default=100.0)
    parser.add_argument("--sequence-positive-window", type=int, default=2)
    parser.add_argument("--sequence-negative-window", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_demo_split(args: argparse.Namespace) -> tuple[List[PlaceRecord], List[PlaceRecord]]:
    load_args = argparse.Namespace(**vars(args))
    load_args.max_database = 0
    load_args.max_queries = 0
    database, queries = load_records(load_args)

    if args.selection_mode == "spread":
        common_count = min(len(database), len(queries))
        start = min(max(0, int(args.database_start)), max(0, common_count - 1))
        needed = max(int(args.database_size), int(args.query_start) + int(args.query_count))
        if needed <= 0:
            raise RuntimeError("--database-size and --query-count must be positive for the demo.")
        base = np.linspace(start, common_count - 1, num=min(needed, common_count - start), dtype=int)
        db_indices = base[: int(args.database_size)]
        query_indices = base[int(args.query_start) : int(args.query_start) + int(args.query_count)]
        database = [database[int(i)] for i in db_indices]
        queries = [queries[int(i)] for i in query_indices]
    else:
        db_start = max(0, int(args.database_start))
        q_start = max(0, int(args.query_start))
        db_end = db_start + int(args.database_size)
        q_end = q_start + int(args.query_count)
        database = database[db_start:db_end]
        queries = queries[q_start:q_end]
    if not database:
        raise RuntimeError("Demo database is empty. Check dataset path and --database-start/--database-size.")
    if not queries:
        raise RuntimeError("Demo query set is empty. Check dataset path and --query-start/--query-count.")
    return database, queries


def public_record_id(record: PlaceRecord, local_idx: int) -> str:
    seq = record.seq_index if record.seq_index is not None else local_idx
    return f"{record.role}_{seq:04d}"


def build_protected_db(db_bits: np.ndarray, database: Sequence[PlaceRecord], args: argparse.Namespace) -> List[ProtectedRecord]:
    protected_db: List[ProtectedRecord] = []
    for db_idx, record in enumerate(database):
        public_id = public_record_id(record, db_idx)
        components: List[ProtectedComponent] = []
        for comp_idx in range(db_bits.shape[1]):
            salt = f"{args.dataset_kind}|{args.gp_database_sequence}|{public_id}|component:{comp_idx}"
            meta = f"record:{public_id}|component:{comp_idx}"
            enrollment = keyed_fe_gen(
                db_bits[db_idx, comp_idx],
                master_secret=args.master_secret,
                salt=salt,
                meta=meta,
                blocks=args.blocks,
                block_width=args.block_width,
                overlap=args.overlap,
                index_seed=2026 + comp_idx,
                max_errors_per_block=args.max_errors_per_block,
                max_total_errors=args.max_total_errors,
                secret_geometry=True,
            )
            components.append(
                ProtectedComponent(
                    helper=enrollment["helper"],
                    tag=enrollment["tag"],
                    meta=meta,
                )
            )
        protected_db.append(ProtectedRecord(public_id=public_id, components=components))
    return protected_db


def scan_record(
    protected_record: ProtectedRecord,
    query_component_bits: np.ndarray,
    *,
    master_secret: str,
) -> int:
    verified_components = 0
    for component in protected_record.components:
        component_ok = False
        for query_code in query_component_bits:
            result = keyed_verify_with_helper(
                query_code,
                component.helper,
                component.tag,
                component.meta,
                master_secret=master_secret,
            )
            if result["verified"]:
                component_ok = True
                break
        if component_ok:
            verified_components += 1
    return verified_components


def expected_positive_ids(
    query: PlaceRecord,
    database: Sequence[PlaceRecord],
    protected_db: Sequence[ProtectedRecord],
    args: argparse.Namespace,
) -> set[str]:
    positive = relation_indices(
        query,
        database,
        positive=True,
        positive_radius_m=args.positive_radius_m,
        negative_radius_m=args.negative_radius_m,
        sequence_positive_window=args.sequence_positive_window,
        sequence_negative_window=args.sequence_negative_window,
    )
    return {protected_db[int(idx)].public_id for idx in positive}


def print_db_state_summary(protected_db: Sequence[ProtectedRecord]) -> None:
    first_helper = protected_db[0].components[0].helper
    helper_keys = ", ".join(sorted(first_helper.keys()))
    print()
    print("Protected DB state")
    print(f"  records: {len(protected_db)}")
    print(f"  components per record: {len(protected_db[0].components)}")
    print("  stored per component: helper, tag, public meta")
    print(f"  helper keys: {helper_keys}")
    print("  open images/descriptors/binary visual codes are not stored in ProtectedRecord")


def main() -> int:
    args = parse_args()

    database, queries = load_demo_split(args)
    db_paths = [record.path for record in database]
    query_paths = [record.path for record in queries]

    print("Protected visual matching demo")
    print(f"  dataset: {args.dataset_kind}")
    print(f"  database sequence: {args.gp_database_sequence}, records: {len(database)}")
    print(f"  query sequence: {args.gp_query_sequence}, queries: {len(queries)}")
    print(f"  acceptance rule: at least {args.min_components} verified components")
    print("  helper geometry: secret")

    print()
    print("Encoding images with DINOv2...")
    encoder = DinoV2TokenEncoder(args.model_name, device=args.device, image_size=args.image_size)
    db_cls, db_tokens = encoder.encode_paths(db_paths, batch_size=args.batch_size)
    query_cls, query_tokens = encoder.encode_paths(query_paths, batch_size=args.batch_size)

    db_components = build_components(db_cls, db_tokens, "regional", args.component_grid)
    query_components = build_components(query_cls, query_tokens, "regional", args.component_grid)

    print("Binarizing transient visual components...")
    db_bits = component_bits(db_components, n_bits=args.component_bits, seed=args.projection_seed)
    query_bits = component_bits(query_components, n_bits=args.component_bits, seed=args.projection_seed)

    print("Creating protected DB records...")
    protected_db = build_protected_db(db_bits, database, args)

    del db_cls, db_tokens, db_components, db_bits
    print_db_state_summary(protected_db)

    print()
    print("Scanning protected DB")
    total_found = 0
    total_true_found = 0
    total_false_accepts = 0

    for query_idx, query in enumerate(queries):
        expected = expected_positive_ids(query, database, protected_db, args)
        scores = []
        for record in protected_db:
            verified = scan_record(record, query_bits[query_idx], master_secret=args.master_secret)
            accepted = verified >= args.min_components
            scores.append((record.public_id, verified, accepted))

        accepted = [(record_id, count) for record_id, count, ok in scores if ok]
        accepted_ids = {record_id for record_id, _count in accepted}
        true_accepts = sorted(accepted_ids & expected)
        false_accepts = sorted(accepted_ids - expected)
        top = sorted(scores, key=lambda row: (-row[1], row[0]))[:5]

        found = bool(accepted)
        true_found = bool(true_accepts)
        total_found += int(found)
        total_true_found += int(true_found)
        total_false_accepts += len(false_accepts)

        print()
        print(f"query {query_idx:02d}: {query.path.name}")
        print(f"  expected same-place ids: {', '.join(sorted(expected)) if expected else 'none'}")
        print(f"  accepted ids: {', '.join(f'{rid}({cnt})' for rid, cnt in accepted) if accepted else 'none'}")
        print(f"  true accepted: {', '.join(true_accepts) if true_accepts else 'none'}")
        print(f"  false accepted: {', '.join(false_accepts) if false_accepts else 'none'}")
        print("  top scanned records:")
        for record_id, count, ok in top:
            marker = "ACCEPT" if ok else "reject"
            print(f"    {record_id}: verified_components={count:02d} {marker}")

    print()
    print("Demo summary")
    print(f"  queries with any accepted record: {total_found}/{len(queries)}")
    print(f"  queries with a true same-place accept: {total_true_found}/{len(queries)}")
    print(f"  false accepted records total: {total_false_accepts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
