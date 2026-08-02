#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from place_data import load_records
from run_place_keyed_component_experiment import (
    component_bits,
    evaluate_pair_set,
    make_helper,
    make_report,
    sample_pairs,
    score_auc,
    write_csv,
)

try:
    from torchvision import transforms
    from torchvision.models import (
        DenseNet121_Weights,
        MobileNet_V3_Large_Weights,
        ResNet50_Weights,
        ViT_B_16_Weights,
        densenet121,
        mobilenet_v3_large,
        resnet50,
        vit_b_16,
    )
except Exception as exc:
    raise RuntimeError("Для компонентного эксперимента с моделями torchvision требуется установленный torchvision") from exc


def _imagenet_preprocess(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class TorchvisionComponentEncoder:
    def __init__(self, model_name: str, *, device: str | None = None, image_size: int = 224, weights: bool = True):
        self.model_name = model_name.lower()
        self.image_size = int(image_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.preprocess = _imagenet_preprocess(self.image_size)
        self.model, self.family, self.out_dim = self._build_model(weights=weights)
        self.model.to(self.device).eval()

    def _build_model(self, *, weights: bool) -> tuple[nn.Module, str, int]:
        if self.model_name == "vit_b_16":
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if weights else None)
            return model, "vit", int(model.hidden_dim)

        if self.model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT if weights else None)
            feature_map = nn.Sequential(*list(model.children())[:-2])
            return feature_map, "conv", int(model.fc.in_features)

        if self.model_name == "densenet121":
            model = densenet121(weights=DenseNet121_Weights.DEFAULT if weights else None)
            return model.features, "densenet", int(model.classifier.in_features)

        if self.model_name == "mobilenet_v3_large":
            model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if weights else None)
            return model.features, "conv", int(model.classifier[0].in_features)

        raise ValueError(
            f"Unsupported model_name={self.model_name!r}. "
            "Use vit_b_16, resnet50, densenet121, or mobilenet_v3_large."
        )

    def _vit_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.model._process_input(x)
        batch = tokens.shape[0]
        cls = self.model.class_token.expand(batch, -1, -1)
        encoded = self.model.encoder(torch.cat([cls, tokens], dim=1))
        cls_out = encoded[:, 0]
        patch_tokens = encoded[:, 1:]
        return F.normalize(cls_out, p=2, dim=1), F.normalize(patch_tokens, p=2, dim=2)

    def _conv_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        fmap = self.model(x)
        if self.family == "densenet":
            fmap = F.relu(fmap, inplace=False)
        if fmap.ndim != 4:
            raise ValueError(f"Expected 4D conv feature map, got shape={tuple(fmap.shape)}")
        return fmap

    @staticmethod
    def _regional_from_tokens(tokens: torch.Tensor, grid: int) -> torch.Tensor:
        n, token_count, dim = tokens.shape
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError(f"Expected square token map, got {token_count} tokens")
        x = tokens.reshape(n, side, side, dim).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(x, output_size=(grid, grid))
        return pooled.permute(0, 2, 3, 1).reshape(n, grid * grid, dim)

    @staticmethod
    def _regional_from_feature_map(fmap: torch.Tensor, grid: int) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(fmap, output_size=(grid, grid))
        return pooled.permute(0, 2, 3, 1).reshape(fmap.shape[0], grid * grid, fmap.shape[1])

    def encode_paths(self, paths: Sequence[Path], *, batch_size: int, component_mode: str, component_grid: int) -> torch.Tensor:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                batch = []
                for path in paths[start : start + batch_size]:
                    image = Image.open(path).convert("RGB")
                    batch.append(self.preprocess(image))
                x = torch.stack(batch, dim=0).to(self.device)

                if self.family == "vit":
                    cls, tokens = self._vit_features(x)
                    if component_mode == "global":
                        comps = cls[:, None, :]
                    elif component_mode == "regional":
                        comps = self._regional_from_tokens(tokens, component_grid)
                    else:
                        raise ValueError(f"Unsupported component_mode={component_mode!r}")
                else:
                    fmap = self._conv_feature_map(x)
                    if component_mode == "global":
                        comps = F.adaptive_avg_pool2d(fmap, output_size=(1, 1)).flatten(1)[:, None, :]
                    elif component_mode == "regional":
                        comps = self._regional_from_feature_map(fmap, component_grid)
                    else:
                        raise ValueError(f"Unsupported component_mode={component_mode!r}")

                chunks.append(F.normalize(comps, p=2, dim=2).cpu())

        if not chunks:
            return torch.empty((0, 0, self.out_dim), dtype=torch.float32)
        return torch.cat(chunks, dim=0)


def parse_args():
    parser = argparse.ArgumentParser(description="Запустить защищенный компонентный эксперимент восстановления для моделей torchvision.")
    parser.add_argument("--dataset-kind", choices=["gardens_point", "vpr_standard", "manifest"], default="gardens_point")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "GardensPointWalking")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--download-gardens-point", action="store_true")
    parser.add_argument("--gp-database-sequence", default="day_right")
    parser.add_argument("--gp-query-sequence", default="night_right")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--model-name", choices=["vit_b_16", "resnet50", "densenet121", "mobilenet_v3_large"], default="vit_b_16")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--component-mode", choices=["global", "regional"], default="regional")
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

    print(f"Вычисление дескрипторов {args.model_name}...")
    encoder = TorchvisionComponentEncoder(args.model_name, image_size=args.image_size, weights=not args.no_pretrained)
    db_components = encoder.encode_paths(
        db_paths,
        batch_size=args.batch_size,
        component_mode=args.component_mode,
        component_grid=args.component_grid,
    )
    query_components = encoder.encode_paths(
        query_paths,
        batch_size=args.batch_size,
        component_mode=args.component_mode,
        component_grid=args.component_grid,
    )
    print(f"Компоненты: база={tuple(db_components.shape)}, запросы={tuple(query_components.shape)}, устройство={encoder.device}")

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
