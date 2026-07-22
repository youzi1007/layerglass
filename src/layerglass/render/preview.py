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
    """Return an RGB image with a random-color mask overlay."""
    out = image.astype(np.float32).copy()
    rng = np.random.default_rng(seed)

    sorted_masks = sorted(
        masks,
        key=lambda annotation: annotation["area"],
        reverse=True,
    )

    for annotation in sorted_masks:
        mask = annotation["segmentation"].astype(bool)
        color = rng.integers(30, 255, size=3).astype(np.float32)

        out[mask] = (
            out[mask] * (1.0 - alpha)
            + color * alpha
        )

    return np.clip(out, 0, 255).astype(np.uint8)


def save_mask_preview(
    image: np.ndarray,
    masks: list[dict],
    output_dir: Path,
) -> None:
    """Save input, overlay, aligned mask arrays, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_masks = sorted(
        masks,
        key=lambda annotation: annotation["area"],
        reverse=True,
    )

    overlay = render_mask_overlay(image, sorted_masks)

    Image.fromarray(overlay).save(output_dir / "mask_overlay.png")
    Image.fromarray(image).save(output_dir / "input.png")

    if sorted_masks:
        mask_stack = np.stack(
            [
                annotation["segmentation"].astype(np.uint8)
                for annotation in sorted_masks
            ],
            axis=0,
        )

        areas = np.asarray(
            [annotation["area"] for annotation in sorted_masks],
            dtype=np.int64,
        )

        bboxes = np.asarray(
            [annotation["bbox"] for annotation in sorted_masks],
            dtype=np.float32,
        )

        predicted_ious = np.asarray(
            [
                annotation.get("predicted_iou", 0.0)
                for annotation in sorted_masks
            ],
            dtype=np.float32,
        )

        stability_scores = np.asarray(
            [
                annotation.get("stability_score", 0.0)
                for annotation in sorted_masks
            ],
            dtype=np.float32,
        )
    else:
        height, width = image.shape[:2]

        mask_stack = np.empty(
            (0, height, width),
            dtype=np.uint8,
        )
        areas = np.empty(0, dtype=np.int64)
        bboxes = np.empty((0, 4), dtype=np.float32)
        predicted_ious = np.empty(0, dtype=np.float32)
        stability_scores = np.empty(0, dtype=np.float32)

    np.savez_compressed(
        output_dir / "masks.npz",
        masks=mask_stack,
        areas=areas,
        bboxes=bboxes,
        predicted_ious=predicted_ious,
        stability_scores=stability_scores,
    )

    summary = []

    for mask_id, annotation in enumerate(sorted_masks):
        summary.append(
            {
                "id": mask_id,
                "area": int(annotation["area"]),
                "bbox_xywh": [
                    float(value)
                    for value in annotation["bbox"]
                ],
                "predicted_iou": float(
                    annotation.get("predicted_iou", 0.0)
                ),
                "stability_score": float(
                    annotation.get("stability_score", 0.0)
                ),
            }
        )

    with open(
        output_dir / "masks_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)
