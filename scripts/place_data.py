#!/usr/bin/env python3
"""Dataset helpers used by the reviewer reproduction scripts."""

from __future__ import annotations

import csv
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

GARDENS_POINT_URL = "https://zenodo.org/records/4590133/files/GardensPointWalking.zip?download=1"


@dataclass(frozen=True)
class PlaceRecord:
    path: Path
    record_id: int
    role: str
    x: Optional[float] = None
    y: Optional[float] = None
    place_id: Optional[str] = None
    seq_index: Optional[int] = None


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and is_image(p))


def maybe_download_gardens_point(dataset_root: Path, download: bool) -> None:
    if any(dataset_root.rglob("*.jpg")) or any(dataset_root.rglob("*.png")):
        return
    if not download:
        return

    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root.parent / "GardensPointWalking.zip"
    if not archive_path.exists():
        print(f"Downloading Gardens Point archive to {archive_path}...")
        urllib.request.urlretrieve(GARDENS_POINT_URL, archive_path)
    print(f"Extracting {archive_path} into {dataset_root}...")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dataset_root)


def normalize_sequence_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def find_sequence_dir(dataset_root: Path, sequence_name: str) -> Path:
    wanted = normalize_sequence_name(sequence_name)
    candidates = [p for p in dataset_root.rglob("*") if p.is_dir()]
    exact = [p for p in candidates if normalize_sequence_name(p.name) == wanted]
    if exact:
        return exact[0]
    contains = [p for p in candidates if wanted in normalize_sequence_name(p.name)]
    if contains:
        return contains[0]
    available = ", ".join(sorted({p.name for p in candidates})[:30])
    raise FileNotFoundError(f"Could not find Gardens Point sequence {sequence_name!r}. Available dirs: {available}")


def sample_paths(paths: List[Path], max_count: int, seed: int) -> List[Path]:
    if max_count <= 0 or len(paths) <= max_count:
        return paths
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(np.arange(len(paths)), size=max_count, replace=False))
    return [paths[int(i)] for i in keep]


def sample_records(records: List[PlaceRecord], max_count: int, seed: int) -> List[PlaceRecord]:
    if max_count <= 0 or len(records) <= max_count:
        return records
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(np.arange(len(records)), size=max_count, replace=False))
    selected = [records[int(i)] for i in keep]
    return [
        PlaceRecord(
            path=r.path,
            record_id=i,
            role=r.role,
            x=r.x,
            y=r.y,
            place_id=r.place_id,
            seq_index=r.seq_index,
        )
        for i, r in enumerate(selected)
    ]


def load_gardens_point_records(args) -> Tuple[List[PlaceRecord], List[PlaceRecord]]:
    maybe_download_gardens_point(args.dataset_root, args.download_gardens_point)
    db_dir = find_sequence_dir(args.dataset_root, args.gp_database_sequence)
    query_dir = find_sequence_dir(args.dataset_root, args.gp_query_sequence)

    db_paths = list_images(db_dir)
    query_paths = list_images(query_dir)
    if not db_paths or not query_paths:
        raise RuntimeError(f"No images found for Gardens Point: database={db_dir}, queries={query_dir}")

    rng = np.random.default_rng(args.seed)
    if args.max_database > 0 and len(db_paths) > args.max_database:
        keep = np.sort(rng.choice(np.arange(len(db_paths)), size=args.max_database, replace=False))
        db_paths = [db_paths[int(i)] for i in keep]
    if args.max_queries > 0 and len(query_paths) > args.max_queries:
        keep = np.sort(rng.choice(np.arange(len(query_paths)), size=args.max_queries, replace=False))
        query_paths = [query_paths[int(i)] for i in keep]

    db_records = [
        PlaceRecord(path=path, record_id=i, role="database", place_id=str(i), seq_index=i)
        for i, path in enumerate(db_paths)
    ]
    query_records = [
        PlaceRecord(path=path, record_id=i, role="query", place_id=str(i), seq_index=i)
        for i, path in enumerate(query_paths)
    ]
    return db_records, query_records


def parse_vpr_standard_filename(path: Path) -> Optional[Tuple[float, float]]:
    parts = path.name.split("@")
    if len(parts) >= 3:
        try:
            return float(parts[1]), float(parts[2])
        except ValueError:
            pass

    numeric = []
    for part in parts:
        try:
            numeric.append(float(part))
        except ValueError:
            continue
        if len(numeric) >= 2:
            return numeric[0], numeric[1]
    return None


def find_vpr_split_dirs(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    base = dataset_root / "images" / split
    db_dir = base / "database"
    query_dir = base / "queries"
    if db_dir.exists() and query_dir.exists():
        return db_dir, query_dir

    db_candidates = [p for p in dataset_root.rglob("database") if p.is_dir() and split in str(p)]
    query_candidates = [p for p in dataset_root.rglob("queries") if p.is_dir() and split in str(p)]
    if db_candidates and query_candidates:
        return db_candidates[0], query_candidates[0]

    raise FileNotFoundError(
        f"Could not find VPR standard split under {dataset_root}. "
        "Expected images/<split>/database and images/<split>/queries."
    )


def load_vpr_standard_records(args) -> Tuple[List[PlaceRecord], List[PlaceRecord]]:
    db_dir, query_dir = find_vpr_split_dirs(args.dataset_root, args.split)
    db_paths = sample_paths(list_images(db_dir), args.max_database, args.seed)
    query_paths = sample_paths(list_images(query_dir), args.max_queries, args.seed + 1)

    db_records = []
    for i, path in enumerate(db_paths):
        coords = parse_vpr_standard_filename(path)
        if coords is not None:
            db_records.append(PlaceRecord(path=path, record_id=i, role="database", x=coords[0], y=coords[1]))

    query_records = []
    for i, path in enumerate(query_paths):
        coords = parse_vpr_standard_filename(path)
        if coords is not None:
            query_records.append(PlaceRecord(path=path, record_id=i, role="query", x=coords[0], y=coords[1]))

    if not db_records or not query_records:
        raise RuntimeError("No coordinate-bearing VPR records found. Check filenames or use --dataset-kind manifest.")
    return db_records, query_records


def load_manifest_records(args) -> Tuple[List[PlaceRecord], List[PlaceRecord]]:
    manifest_path = args.manifest
    if manifest_path is None:
        raise ValueError("--manifest is required for --dataset-kind manifest")

    base_dir = args.dataset_root if args.dataset_root else manifest_path.parent
    database: List[PlaceRecord] = []
    queries: List[PlaceRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = (row.get("role") or row.get("split") or "").strip().lower()
            if role in {"db", "reference", "database"}:
                role = "database"
            elif role in {"q", "query", "queries"}:
                role = "query"
            else:
                continue

            raw_path = row.get("path") or row.get("image") or row.get("filepath")
            if not raw_path:
                continue
            path = Path(raw_path)
            if not path.is_absolute():
                path = base_dir / path

            def maybe_float(name: str) -> Optional[float]:
                value = row.get(name)
                if value in {None, ""}:
                    return None
                return float(value)

            record = PlaceRecord(
                path=path,
                record_id=len(database) if role == "database" else len(queries),
                role=role,
                x=maybe_float("x") if row.get("x") not in {None, ""} else maybe_float("utm_e"),
                y=maybe_float("y") if row.get("y") not in {None, ""} else maybe_float("utm_n"),
                place_id=(row.get("place_id") or row.get("label") or None),
                seq_index=int(row["seq_index"]) if row.get("seq_index") not in {None, ""} else None,
            )
            if role == "database":
                database.append(record)
            else:
                queries.append(record)

    database = sample_records(database, args.max_database, args.seed)
    queries = sample_records(queries, args.max_queries, args.seed + 1)
    if not database or not queries:
        raise RuntimeError("Manifest did not produce non-empty database/query records.")
    return database, queries


def relation_indices(
    query: PlaceRecord,
    database: Sequence[PlaceRecord],
    *,
    positive: bool,
    positive_radius_m: float,
    negative_radius_m: float,
    sequence_positive_window: int,
    sequence_negative_window: int,
) -> np.ndarray:
    idxs = []
    for db_idx, db in enumerate(database):
        if query.x is not None and query.y is not None and db.x is not None and db.y is not None:
            d = math.hypot(float(query.x) - float(db.x), float(query.y) - float(db.y))
            if positive and d <= positive_radius_m:
                idxs.append(db_idx)
            elif not positive and d >= negative_radius_m:
                idxs.append(db_idx)
            continue

        if query.seq_index is not None and db.seq_index is not None:
            d = abs(int(query.seq_index) - int(db.seq_index))
            if positive and d <= sequence_positive_window:
                idxs.append(db_idx)
            elif not positive and d >= sequence_negative_window:
                idxs.append(db_idx)
            continue

        if query.place_id is not None and db.place_id is not None:
            same = query.place_id == db.place_id
            if positive and same:
                idxs.append(db_idx)
            elif not positive and not same:
                idxs.append(db_idx)
    return np.asarray(idxs, dtype=np.int64)


def load_records(args) -> Tuple[List[PlaceRecord], List[PlaceRecord]]:
    if args.dataset_kind == "gardens_point":
        return load_gardens_point_records(args)
    if args.dataset_kind == "vpr_standard":
        return load_vpr_standard_records(args)
    if args.dataset_kind == "manifest":
        return load_manifest_records(args)
    raise ValueError(f"Unsupported dataset_kind={args.dataset_kind!r}")
