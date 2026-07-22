from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


MODEL_CONFIGS = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
}


@dataclass
class DepthConfig:
    encoder: str = "vits"
    input_size: int = 518
    repository: Path = Path(
        "~/projects/Depth-Anything-V2"
    ).expanduser()
    checkpoint: Path = Path(
        "~/projects/Depth-Anything-V2/checkpoints/"
        "depth_anything_v2_vits.pth"
    ).expanduser()


def build_depth_model(config: DepthConfig) -> torch.nn.Module:
    if config.encoder not in MODEL_CONFIGS:
        raise ValueError(
            f"Unsupported encoder {config.encoder!r}. "
            f"Choose from: {', '.join(MODEL_CONFIGS)}"
        )

    if not config.repository.exists():
        raise FileNotFoundError(
            f"Depth Anything repository not found: {config.repository}"
        )

    if not config.checkpoint.exists():
        raise FileNotFoundError(
            f"Depth checkpoint not found: {config.checkpoint}"
        )

    repository_string = str(config.repository)
    if repository_string not in sys.path:
        sys.path.insert(0, repository_string)

    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(**MODEL_CONFIGS[config.encoder])

    state_dict = torch.load(
        config.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    return model.to(device).eval()


def infer_depth(
    image_path: Path,
    config: DepthConfig | None = None,
) -> np.ndarray:
    config = config or DepthConfig()

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    model = build_depth_model(config)

    with torch.inference_mode():
        depth = model.infer_image(
            image,
            input_size=config.input_size,
        )

    return np.asarray(depth, dtype=np.float32)


def save_depth_outputs(
    depth: np.ndarray,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "depth.npy", depth)

    depth_min = float(depth.min())
    depth_max = float(depth.max())

    if depth_max <= depth_min:
        preview = np.zeros(depth.shape, dtype=np.uint8)
    else:
        normalized = (
            (depth - depth_min) / (depth_max - depth_min)
        )
        preview = np.round(normalized * 255).astype(np.uint8)

    cv2.imwrite(
        str(output_dir / "depth_preview.png"),
        preview,
    )
