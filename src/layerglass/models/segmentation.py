from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


@dataclass
class Sam2MaskConfig:
    checkpoint: Path = Path("~/projects/sam2/checkpoints/sam2.1_hiera_large.pt").expanduser()
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"

    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    crop_n_layers: int = 1
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: int = 400


def load_rgb_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def build_generator(config: Sam2MaskConfig) -> SAM2AutomaticMaskGenerator:
    if not config.checkpoint.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {config.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_sam2(
        config.model_cfg,
        str(config.checkpoint),
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
        use_m2m=True,
    )


def generate_masks(image: np.ndarray, config: Sam2MaskConfig | None = None) -> list[dict]:
    config = config or Sam2MaskConfig()
    generator = build_generator(config)
    return generator.generate(image)
