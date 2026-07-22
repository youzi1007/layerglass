from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


MODEL_CONFIGS = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

MODEL_CHECKPOINTS = {
    "tiny": Path("~/projects/sam2/checkpoints/sam2.1_hiera_tiny.pt").expanduser(),
    "small": Path("~/projects/sam2/checkpoints/sam2.1_hiera_small.pt").expanduser(),
    "base_plus": Path("~/projects/sam2/checkpoints/sam2.1_hiera_base_plus.pt").expanduser(),
    "large": Path("~/projects/sam2/checkpoints/sam2.1_hiera_large.pt").expanduser(),
}


@dataclass
class Sam2MaskConfig:
    model_size: str = "small"
    checkpoint: Path | None = None
    model_cfg: str | None = None

    points_per_side: int = 24
    points_per_batch: int = 16
    pred_iou_thresh: float = 0.78
    stability_score_thresh: float = 0.86
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: int = 1200
    use_m2m: bool = False
    max_side_px: int = 1400


def load_rgb_image(path: Path, max_side_px: int | None = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    arr = np.array(image)

    if max_side_px is None:
        return arr

    h, w = arr.shape[:2]
    scale = max_side_px / max(h, w)

    if scale >= 1.0:
        return arr

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def build_generator(config: Sam2MaskConfig) -> SAM2AutomaticMaskGenerator:
    if config.model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model_size={config.model_size!r}. "
            f"Choose one of: {', '.join(MODEL_CONFIGS)}"
        )

    checkpoint = config.checkpoint or MODEL_CHECKPOINTS[config.model_size]
    model_cfg = config.model_cfg or MODEL_CONFIGS[config.model_size]

    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_sam2(
        model_cfg,
        str(checkpoint),
        device=device,
        apply_postprocessing=False,
    )

    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=config.points_per_side,
        points_per_batch=config.points_per_batch,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_score_thresh,
        crop_n_layers=config.crop_n_layers,
        crop_n_points_downscale_factor=config.crop_n_points_downscale_factor,
        min_mask_region_area=config.min_mask_region_area,
        use_m2m=config.use_m2m,
    )


def generate_masks(image: np.ndarray, config: Sam2MaskConfig | None = None) -> list[dict]:
    config = config or Sam2MaskConfig()
    generator = build_generator(config)
    return generator.generate(image)
