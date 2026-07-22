from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def render_mask_overlay(
    image: np.ndarray,
    masks: list[dict],
    alpha: float = 0.5,
    seed: int = 7,
) -> np.ndarray:
    """Return RGB image with random-color mask overlay."""
    out = image.astype(np.float32).copy()
    rng = np.random.default_rng(seed)

    # Draw large regions first, small regions on top.
    sorted_masks = sorted(masks, key=lambda m: m["area"], reverse=True)

    for ann in sorted_masks:
        mask = ann["segmentation"].astype(bool)
        color = rng.integers(30, 255, size=3).astype(np.float32)
        out[mask] = out[mask] * (1.0 - alpha) + color * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def save_mask_preview(
    image: np.ndarray,
    masks: list[dict],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = render_mask_overlay(image, masks)
    Image.fromarray(overlay).save(output_dir / "mask_overlay.png")
    Image.fromarray(image).save(output_dir / "input.png")

    summary = []
    for i, ann in enumerate(sorted(masks, key=lambda m: m["area"], reverse=True)):
        summary.append(
            {
                "id": i,
                "area": int(ann["area"]),
                "bbox_xywh": [float(x) for x in ann["bbox"]],
                "predicted_iou": float(ann.get("predicted_iou", 0.0)),
                "stability_score": float(ann.get("stability_score", 0.0)),
            }
        )

    with open(output_dir / "masks_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
