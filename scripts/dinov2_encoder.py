#!/usr/bin/env python3
"""Small DINOv2 token encoder used by the protected matching scripts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision import transforms
except Exception as exc:
    raise RuntimeError("torchvision is required for DINOv2 preprocessing") from exc


DINOV2_COMMIT = "7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c"
DINOV2_REPOSITORY = f"facebookresearch/dinov2:{DINOV2_COMMIT}"
DINOV2_WEIGHT_SHA256 = {
    "dinov2_vits14": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_cached_weights(model_name: str) -> None:
    expected = DINOV2_WEIGHT_SHA256.get(model_name)
    if expected is None:
        return
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / f"{model_name}_pretrain.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"DINOv2 checkpoint was not cached after model loading: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise RuntimeError(
            f"DINOv2 checkpoint SHA-256 mismatch for {checkpoint.name}: "
            f"expected {expected}, got {actual}"
        )


class DinoV2TokenEncoder:
    def __init__(self, model_name: str, device: str | None = None, image_size: int = 224):
        self.model_name = model_name
        self.image_size = int(image_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load(
            DINOV2_REPOSITORY,
            model_name,
            trust_repo=True,
            skip_validation=True,
        ).to(self.device).eval()
        verify_cached_weights(model_name)
        print(f"DINOv2 source commit: {DINOV2_COMMIT}")
        if model_name in DINOV2_WEIGHT_SHA256:
            print(f"DINOv2 weights SHA-256: {DINOV2_WEIGHT_SHA256[model_name]}")
        print(f"DINOv2 device: {self.device}")
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def encode_paths(self, paths: Sequence[Path], batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        cls_chunks = []
        token_chunks = []
        with torch.inference_mode():
            for start in range(0, len(paths), batch_size):
                batch = []
                for path in paths[start : start + batch_size]:
                    image = Image.open(path).convert("RGB")
                    batch.append(self.preprocess(image))
                x = torch.stack(batch, dim=0).to(self.device)
                features = self.model.forward_features(x)
                cls = F.normalize(features["x_norm_clstoken"], p=2, dim=1)
                tokens = F.normalize(features["x_norm_patchtokens"], p=2, dim=2)
                cls_chunks.append(cls.cpu())
                token_chunks.append(tokens.cpu())
        return torch.cat(cls_chunks, dim=0), torch.cat(token_chunks, dim=0)
