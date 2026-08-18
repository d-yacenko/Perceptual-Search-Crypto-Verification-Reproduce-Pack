#!/usr/bin/env python3
import argparse
import csv
import hmac
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from perceptual_crypto import derive_keyed_secret_bits, derive_keyed_seed, fe_block_indices, keyed_tag_from_secret_bits
from place_data import load_records, relation_indices
from dinov2_encoder import DinoV2TokenEncoder


@dataclass
class FastKeyedHelper:
    idx: np.ndarray
    p_bits: np.ndarray
    salt: str
    tag: str
    meta: str
    blocks: int
    block_width: int
    max_errors_per_block: int
    max_total_errors: int
    public_geometry: bool


def regional_components(tokens: torch.Tensor, grid: int) -> torch.Tensor:
    n, token_count, dim = tokens.shape
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise ValueError(f"Expected square token map, got {token_count} tokens")
    if side % grid != 0:
        raise ValueError(f"Token side {side} must be divisible by grid={grid}")

    cell = side // grid
    x = tokens.reshape(n, side, side, dim)
    comps = []
    for gy in range(grid):
        for gx in range(grid):
            patch = x[:, gy * cell : (gy + 1) * cell, gx * cell : (gx + 1) * cell, :]
            comps.append(patch.mean(dim=(1, 2)))
    out = torch.stack(comps, dim=1)
    return F.normalize(out, p=2, dim=2)


def build_components(cls: torch.Tensor, tokens: torch.Tensor, mode: str, grid: int) -> torch.Tensor:
    if mode == "cls":
        return cls[:, None, :]
    if mode == "regional":
        return regional_components(tokens, grid=grid)
    raise ValueError(f"Unsupported component mode: {mode}")


def component_bits(components: torch.Tensor, *, n_bits: int, seed: int) -> np.ndarray:
    comps = components.cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n_bits, comps.shape[-1])).astype(np.float32)
    z = np.einsum("ncd,bd->ncb", comps, a, optimize=True)
    return (z >= 0).astype(np.uint8)


def make_helper(
    code: np.ndarray,
    *,
    master_secret: str,
    salt: str,
    blocks: int,
    block_width: int,
    overlap: int,
    index_seed: int,
    max_errors_per_block: int,
    max_total_errors: int,
    meta: str,
    secret_geometry: bool,
) -> FastKeyedHelper:
    effective_index_seed = derive_keyed_seed(master_secret, salt, "index-seed") if secret_geometry else index_seed
    idx = fe_block_indices(len(code), blocks, block_width, overlap=overlap, index_seed=effective_index_seed)
    expected = derive_keyed_secret_bits(master_secret, salt, blocks)
    code_matrix = np.repeat(expected[:, None], block_width, axis=1).astype(np.uint8)
    p_bits = (code[idx] ^ code_matrix).astype(np.uint8)
    tag = keyed_tag_from_secret_bits(expected, master_secret, salt, meta)
    return FastKeyedHelper(
        idx=idx,
        p_bits=p_bits,
        salt=salt,
        tag=tag,
        meta=meta,
        blocks=blocks,
        block_width=block_width,
        max_errors_per_block=max_errors_per_block,
        max_total_errors=max_total_errors,
        public_geometry=not secret_geometry,
    )


def decode_component(code: np.ndarray, helper: FastKeyedHelper) -> Tuple[np.ndarray, bool, int, int, int, int]:
    observed = (code[helper.idx] ^ helper.p_bits).astype(np.uint8)
    ones = observed.sum(axis=1)
    secret_hat = (ones >= ((helper.block_width + 1) // 2)).astype(np.uint8)
    corrected_per_block = np.minimum(ones, helper.block_width - ones).astype(np.int64)
    corrected_total = int(corrected_per_block.sum())
    max_corrected = int(corrected_per_block.max())
    overfull = int(np.sum(corrected_per_block > helper.max_errors_per_block))
    decode_ok = (overfull == 0) and (corrected_total <= helper.max_total_errors)
    penalty = corrected_total + max_corrected * helper.blocks
    return secret_hat, bool(decode_ok), int(penalty), corrected_total, max_corrected, overfull


def verify_component(code: np.ndarray, helper: FastKeyedHelper, *, master_secret: str) -> Tuple[bool, int]:
    secret_hat, decode_ok, penalty, _corrected_total, _max_corrected, _overfull = decode_component(code, helper)
    if not decode_ok:
        return False, penalty

    tag_q = keyed_tag_from_secret_bits(secret_hat, master_secret, helper.salt, helper.meta)
    tag_ok = hmac.compare_digest(tag_q, helper.tag)
    return bool(tag_ok), penalty


def helper_only_component(code: np.ndarray, helper: FastKeyedHelper) -> Tuple[bool, int]:
    if not helper.public_geometry:
        raise ValueError("helper-only alignment score is unavailable for secret geometry")
    _secret_hat, decode_ok, penalty, _corrected_total, _max_corrected, _overfull = decode_component(code, helper)
    return bool(decode_ok), penalty


def match_components(
    db_helpers: Sequence[FastKeyedHelper],
    query_codes: np.ndarray,
    *,
    master_secret: str,
) -> Tuple[int, Optional[int], Optional[int]]:
    helper_only_available = all(helper.public_geometry for helper in db_helpers)
    verified_count = 0
    helper_only_count = 0
    best_penalty = 10**12
    used_verified_query_components = set()
    used_helper_only_query_components = set()
    for helper in db_helpers:
        component_verified = False
        component_helper_only = False
        component_best_verified_penalty = 10**12
        component_best_helper_only_penalty = 10**12
        component_verified_q = None
        component_helper_only_q = None
        for q_idx, q_code in enumerate(query_codes):
            verified, verified_penalty = verify_component(q_code, helper, master_secret=master_secret)
            if helper_only_available:
                helper_only, helper_only_penalty = helper_only_component(q_code, helper)
            else:
                helper_only, helper_only_penalty = False, None
            if verified_penalty < component_best_verified_penalty:
                component_best_verified_penalty = verified_penalty
                component_verified_q = q_idx
            if helper_only_penalty is not None and helper_only_penalty < component_best_helper_only_penalty:
                component_best_helper_only_penalty = helper_only_penalty
                component_helper_only_q = q_idx
            if helper_only:
                component_helper_only = True
                if q_idx not in used_helper_only_query_components:
                    component_helper_only_q = q_idx
            if verified:
                component_verified = True
                if q_idx not in used_verified_query_components:
                    component_verified_q = q_idx
                    break
        if component_verified:
            verified_count += 1
            if component_verified_q is not None:
                used_verified_query_components.add(component_verified_q)
        if helper_only_available and component_helper_only:
            helper_only_count += 1
            if component_helper_only_q is not None:
                used_helper_only_query_components.add(component_helper_only_q)
        if helper_only_available:
            best_penalty = min(best_penalty, component_best_helper_only_penalty)
    if not helper_only_available:
        return verified_count, None, None
    return verified_count, helper_only_count, int(best_penalty)


def sample_pairs(database, queries, *, positive: bool, max_pairs: int, args, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    pairs = []
    seen = set()
    query_order = np.arange(len(queries))
    rng.shuffle(query_order)

    for query_idx in query_order:
        candidates = relation_indices(
            queries[int(query_idx)],
            database,
            positive=positive,
            positive_radius_m=args.positive_radius_m,
            negative_radius_m=args.negative_radius_m,
            sequence_positive_window=args.sequence_positive_window,
            sequence_negative_window=args.sequence_negative_window,
        )
        if len(candidates) == 0:
            continue
        pair = (int(rng.choice(candidates)), int(query_idx))
        pairs.append(pair)
        seen.add(pair)
        if len(pairs) >= max_pairs:
            return pairs

    attempts = 0
    while len(pairs) < max_pairs and attempts < max_pairs * 100:
        attempts += 1
        query_idx = int(rng.integers(0, len(queries)))
        candidates = relation_indices(
            queries[query_idx],
            database,
            positive=positive,
            positive_radius_m=args.positive_radius_m,
            negative_radius_m=args.negative_radius_m,
            sequence_positive_window=args.sequence_positive_window,
            sequence_negative_window=args.sequence_negative_window,
        )
        if len(candidates) == 0:
            continue
        pair = (int(rng.choice(candidates)), query_idx)
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return pairs


def summarize(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": math.nan, "q50": math.nan, "q90": math.nan, "q95": math.nan, "max": math.nan}
    return {
        "mean": float(arr.mean()),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def score_auc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return math.nan
    cmp = pos[:, None] - neg[None, :]
    return float((np.sum(cmp > 0) + 0.5 * np.sum(cmp == 0)) / cmp.size)


def evaluate_pair_set(name: str, pairs: Sequence[Tuple[int, int]], db_helpers, query_bits, *, master_secret: str) -> dict:
    counts = []
    helper_only_counts = []
    penalties = []
    for db_idx, query_idx in pairs:
        count, helper_only_count, penalty = match_components(db_helpers[db_idx], query_bits[query_idx], master_secret=master_secret)
        counts.append(count)
        if helper_only_count is not None:
            helper_only_counts.append(helper_only_count)
        if penalty is not None:
            penalties.append(penalty)
    count_summary = summarize(counts)
    helper_only_summary = summarize(helper_only_counts)
    penalty_summary = summarize(penalties)
    return {
        "pair_set": name,
        "n_pairs": len(pairs),
        "match_counts": np.asarray(counts, dtype=np.int64),
        "helper_only_counts": np.asarray(helper_only_counts, dtype=np.int64),
        "penalties": np.asarray(penalties, dtype=np.int64),
        "match_count_mean": count_summary["mean"],
        "match_count_q50": count_summary["q50"],
        "match_count_q90": count_summary["q90"],
        "match_count_q95": count_summary["q95"],
        "match_count_max": count_summary["max"],
        "helper_only_count_mean": helper_only_summary["mean"],
        "helper_only_count_q50": helper_only_summary["q50"],
        "helper_only_count_q90": helper_only_summary["q90"],
        "helper_only_count_q95": helper_only_summary["q95"],
        "helper_only_count_max": helper_only_summary["max"],
        "penalty_mean": penalty_summary["mean"],
        "penalty_q50": penalty_summary["q50"],
        "penalty_q90": penalty_summary["q90"],
    }


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_report(path: Path, *, args, summary_rows: List[dict], threshold_rows: List[dict], leakage_rows: List[dict]) -> None:
    def fmt(value: object, digits: int) -> str:
        number = float(value)
        return "N/A" if math.isnan(number) else f"{number:.{digits}f}"

    pair_labels = {
        "positive_place": "одно место",
        "negative_place": "разные места",
    }
    lines = [
        "# Защищенное компонентное сопоставление с восстановлением по вспомогательным данным",
        "",
        "Отчет оценивает защищенное сопоставление по вспомогательным данным и тегам без хранения открытых векторных представлений или бинарных визуальных отпечатков в базе.",
        "",
        "## Конфигурация",
        "",
        f"- Тип набора данных: `{args.dataset_kind}`",
        f"- Последовательность базы: `{args.gp_database_sequence}`",
        f"- Последовательность запросов: `{args.gp_query_sequence}`",
        f"- Модель: `{args.model_name}`",
        f"- Режим компонент: `{args.component_mode}`",
        f"- Сетка компонент: `{args.component_grid}`",
        f"- Битов на компоненту: `{args.component_bits}`",
        f"- Компонент на изображение: `{summary_rows[0]['components_per_image'] if summary_rows else ''}`",
        f"- Блоки/ширина восстановления: `{args.blocks}` / `{args.block_width}`",
        f"- Максимум ошибок на блок / суммарно: `{args.max_errors_per_block}` / `{args.max_total_errors}`",
        f"- Окно совпадающих/несовпадающих позиций в последовательности: `{args.sequence_positive_window}` / `{args.sequence_negative_window}`",
        f"- Радиус совпадающих/несовпадающих пар, м: `{args.positive_radius_m}` / `{args.negative_radius_m}`",
        f"- Секретная геометрия вспомогательных данных: `{args.secret_geometry}`",
        "",
        "## Число подтвержденных компонент",
        "",
        "| множество пар | n | среднее подтвержденных | среднее по вспомогательным данным | q90 проверки | q90 по вспомогательным данным | максимум проверки | максимум по вспомогательным данным |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        pair_set = pair_labels.get(str(row["pair_set"]), row["pair_set"])
        lines.append(
            f"| {pair_set} | {row['n_pairs']} | {fmt(row['match_count_mean'], 3)} | "
            f"{fmt(row['helper_only_count_mean'], 3)} | {fmt(row['match_count_q90'], 3)} | "
            f"{fmt(row['helper_only_count_q90'], 3)} | {fmt(row['match_count_max'], 3)} | "
            f"{fmt(row['helper_only_count_max'], 3)} |"
        )

    lines += [
        "",
        "## Перебор порога",
        "",
        "| минимум компонент | TPR проверки | FAR проверки | FRR проверки | TPR по вспомогательным данным | FAR по вспомогательным данным | FRR по вспомогательным данным |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in threshold_rows:
        lines.append(
            f"| {row['min_verified_components']} | {fmt(row['tpr'], 4)} | "
            f"{fmt(row['far'], 4)} | {fmt(row['frr'], 4)} | "
            f"{fmt(row['helper_only_tpr'], 4)} | {fmt(row['helper_only_far'], 4)} | "
            f"{fmt(row['helper_only_frr'], 4)} |"
        )
    lines += [
        "",
        "## Утечка через вспомогательные данные",
        "",
        "В этом разделе вспомогательные данные считаются публичными, а главный секретный ключ неизвестен.",
        "Если геометрия публична, атакующий не может проверить HMAC-тег, но может",
        "оценить, близок ли XOR между кодом запроса и `helper_P` к константному блоку повторения.",
        "Если геометрия секретна, такое прямое применение вспомогательных данных",
        "в данном эксперименте недоступно атакующему.",
        "",
        "| метрика | значение |",
        "|---|---:|",
    ]
    for row in leakage_rows:
        lines.append(f"| {row['metric']} | {fmt(row['value'], 6)} |")
    lines += [
        "",
        "## Примечания",
        "",
        "- `TPR` -- доля пар одного места, прошедших порог по числу компонент.",
        "- `FAR` -- доля удаленных пар или пар разных мест, прошедших тот же порог.",
        "- Компонентная проверка использует сохраненный тег; предполагаемое состояние базы содержит только соль, вспомогательные данные, тег и публичные параметры политики.",
        "- При публичной геометрии метрики по одним вспомогательным данным оценивают автономный сигнал проверки принадлежности.",
        "- При секретной геометрии прямой тест выравнивания не вычисляется без секретных позиций, поэтому соответствующие метрики обозначаются как N/A.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Запустить защищенный компонентный эксперимент восстановления для VPR.")
    parser.add_argument("--dataset-kind", choices=["gardens_point", "vpr_standard", "manifest"], default="gardens_point")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "GardensPointWalking")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--download-gardens-point", action="store_true")
    parser.add_argument("--gp-database-sequence", default="day_right")
    parser.add_argument("--gp-query-sequence", default="night_right")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--model-name", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    parser.add_argument("--device", default="cpu", help="PyTorch device; CPU is the canonical reproducibility mode.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--component-mode", choices=["cls", "regional"], default="regional")
    parser.add_argument("--component-grid", type=int, default=4)
    parser.add_argument("--component-bits", type=int, default=4096)
    parser.add_argument("--projection-seed", type=int, default=42)
    parser.add_argument("--master-secret", default="article18-demo-master-secret-32-bytes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-radius-m", type=float, default=25.0)
    parser.add_argument("--negative-radius-m", type=float, default=100.0)
    parser.add_argument("--sequence-positive-window", type=int, default=2)
    parser.add_argument("--sequence-negative-window", type=int, default=20)
    parser.add_argument("--max-database", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--max-positive-pairs", type=int, default=400)
    parser.add_argument("--max-negative-pairs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--block-width", type=int, default=41)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--max-errors-per-block", type=int, default=16)
    parser.add_argument("--max-total-errors", type=int, default=1000)
    parser.add_argument("--min-components", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    parser.add_argument("--secret-geometry", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    database, queries = load_records(args)
    print(f"Набор данных: {args.dataset_kind}, база={len(database)}, запросы={len(queries)}")
    db_paths = [record.path for record in database]
    query_paths = [record.path for record in queries]

    print("Вычисление дескрипторов DINOv2...")
    encoder = DinoV2TokenEncoder(args.model_name, device=args.device, image_size=args.image_size)
    db_cls, db_tokens = encoder.encode_paths(db_paths, batch_size=args.batch_size)
    query_cls, query_tokens = encoder.encode_paths(query_paths, batch_size=args.batch_size)

    db_components = build_components(db_cls, db_tokens, args.component_mode, args.component_grid)
    query_components = build_components(query_cls, query_tokens, args.component_mode, args.component_grid)
    print(f"Компоненты: база={tuple(db_components.shape)}, запросы={tuple(query_components.shape)}")

    print("Бинаризация компонент...")
    db_bits = component_bits(db_components, n_bits=args.component_bits, seed=args.projection_seed)
    query_bits = component_bits(query_components, n_bits=args.component_bits, seed=args.projection_seed)

    print("Построение вспомогательных данных с секретным ключом в памяти...")
    db_helpers = []
    for db_idx in range(len(database)):
        row_helpers = []
        for comp_idx in range(db_bits.shape[1]):
            salt = f"{args.dataset_kind}|{args.gp_database_sequence}|db:{db_idx}|comp:{comp_idx}"
            meta = f"{args.dataset_kind}|{args.gp_database_sequence}|record:{db_idx}|component:{comp_idx}"
            row_helpers.append(
                make_helper(
                    db_bits[db_idx, comp_idx],
                    master_secret=args.master_secret,
                    salt=salt,
                    blocks=args.blocks,
                    block_width=args.block_width,
                    overlap=args.overlap,
                    index_seed=2026 + comp_idx,
                    max_errors_per_block=args.max_errors_per_block,
                    max_total_errors=args.max_total_errors,
                    meta=meta,
                    secret_geometry=args.secret_geometry,
                )
            )
        db_helpers.append(row_helpers)

    positive_pairs = sample_pairs(database, queries, positive=True, max_pairs=args.max_positive_pairs, args=args, seed=args.seed + 1)
    negative_pairs = sample_pairs(database, queries, positive=False, max_pairs=args.max_negative_pairs, args=args, seed=args.seed + 2)
    print(f"Пары: совпадающие={len(positive_pairs)}, несовпадающие={len(negative_pairs)}")

    print("Оценка компонентного восстановления на множествах пар...")
    pos_stats = evaluate_pair_set("positive_place", positive_pairs, db_helpers, query_bits, master_secret=args.master_secret)
    neg_stats = evaluate_pair_set("negative_place", negative_pairs, db_helpers, query_bits, master_secret=args.master_secret)

    components_per_image = int(db_bits.shape[1])
    summary_rows = []
    for stats in [pos_stats, neg_stats]:
        row = {k: v for k, v in stats.items() if k not in {"match_counts", "helper_only_counts", "penalties"}}
        row["components_per_image"] = components_per_image
        summary_rows.append(row)

    threshold_rows = []
    for threshold in args.min_components:
        tpr = float((pos_stats["match_counts"] >= threshold).mean()) if len(pos_stats["match_counts"]) else math.nan
        far = float((neg_stats["match_counts"] >= threshold).mean()) if len(neg_stats["match_counts"]) else math.nan
        helper_only_tpr = float((pos_stats["helper_only_counts"] >= threshold).mean()) if len(pos_stats["helper_only_counts"]) else math.nan
        helper_only_far = float((neg_stats["helper_only_counts"] >= threshold).mean()) if len(neg_stats["helper_only_counts"]) else math.nan
        threshold_rows.append(
            {
                "min_verified_components": int(threshold),
                "tpr": tpr,
                "far": far,
                "frr": 1.0 - tpr if not math.isnan(tpr) else math.nan,
                "helper_only_tpr": helper_only_tpr,
                "helper_only_far": helper_only_far,
                "helper_only_frr": 1.0 - helper_only_tpr if not math.isnan(helper_only_tpr) else math.nan,
            }
        )

    leakage_rows = [
        {
            "metric": "helper_only_component_count_auc",
            "value": score_auc(pos_stats["helper_only_counts"], neg_stats["helper_only_counts"]),
        },
        {
            "metric": "helper_only_best_penalty_auc",
            "value": score_auc(-pos_stats["penalties"], -neg_stats["penalties"]),
        },
        {
            "metric": "verified_component_count_auc",
            "value": score_auc(pos_stats["match_counts"], neg_stats["match_counts"]),
        },
    ]

    dataset_tag = f"{args.dataset_kind}_{args.gp_database_sequence}_{args.gp_query_sequence}"
    component_tag = args.component_mode
    if args.component_mode == "regional":
        component_tag = f"{component_tag}g{args.component_grid}"
    prefix = (
        f"protected_{dataset_tag}_{args.model_name}_{component_tag}_"
        f"bits{args.component_bits}_b{args.blocks}_w{args.block_width}_"
        f"o{args.overlap}_mb{args.max_errors_per_block}_mt{args.max_total_errors}_"
        f"sg{int(args.secret_geometry)}"
    )
    summary_fields = [
        "pair_set",
        "n_pairs",
        "components_per_image",
        "match_count_mean",
        "match_count_q50",
        "match_count_q90",
        "match_count_q95",
        "match_count_max",
        "helper_only_count_mean",
        "helper_only_count_q50",
        "helper_only_count_q90",
        "helper_only_count_q95",
        "helper_only_count_max",
        "penalty_mean",
        "penalty_q50",
        "penalty_q90",
    ]
    threshold_fields = [
        "min_verified_components",
        "tpr",
        "far",
        "frr",
        "helper_only_tpr",
        "helper_only_far",
        "helper_only_frr",
    ]
    leakage_fields = ["metric", "value"]
    write_csv(args.reports_dir / f"{prefix}_pair_summary.csv", summary_rows, summary_fields)
    write_csv(args.reports_dir / f"{prefix}_threshold_sweep.csv", threshold_rows, threshold_fields)
    write_csv(args.reports_dir / f"{prefix}_leakage_summary.csv", leakage_rows, leakage_fields)
    print(f"Отчеты записаны с префиксом {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
