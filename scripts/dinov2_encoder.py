#!/usr/bin/env python3
"""Small DINOv2 token encoder used by the protected matching scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchvision import transforms
except Exception as exc:
    raise RuntimeError("torchvision is required for DINOv2 preprocessing") from exc


class DinoV2TokenEncoder:
    def __init__(self, model_name: str, device: str | None = None, image_size: int = 224):
        self.model_name = model_name
        self.image_size = int(image_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True).to(self.device).eval()
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
