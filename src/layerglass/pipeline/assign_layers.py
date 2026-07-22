from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class MaskAnalysisConfig:
    """Settings for mask scoring and duplicate removal."""

    min_area_px: int = 100
    duplicate_iou: float = 0.90
    duplicate_containment: float = 0.98
    duplicate_area_ratio: float = 0.80
    overlay_alpha: float = 0.48


def _compute_bbox(mask: np.ndarray) -> np.ndarray:
    y_values, x_values = np.nonzero(mask)

    if len(x_values) == 0:
        return np.asarray([0, 0, 0, 0], dtype=np.float32)

    x_min = int(x_values.min())
    x_max = int(x_values.max()) + 1
    y_min = int(y_values.min())
    y_max = int(y_values.max()) + 1

    return np.asarray(
        [
            x_min,
            y_min,
            x_max - x_min,
            y_max - y_min,
        ],
        dtype=np.float32,
    )


def load_mask_candidates(
    masks_path: Path,
) -> dict[str, np.ndarray]:
    """Load masks and their SAM quality metadata."""
    data = np.load(masks_path)

    masks = data["masks"].astype(bool)
    mask_count = masks.shape[0]

    if "areas" in data.files:
        areas = data["areas"].astype(np.int64)
    else:
        areas = masks.sum(axis=(1, 2)).astype(np.int64)

    if "bboxes" in data.files:
        bboxes = data["bboxes"].astype(np.float32)
    else:
        bboxes = np.stack(
            [_compute_bbox(mask) for mask in masks],
            axis=0,
        )

    if "predicted_ious" in data.files:
        predicted_ious = data["predicted_ious"].astype(
            np.float32
        )
    else:
        predicted_ious = np.zeros(
            mask_count,
            dtype=np.float32,
        )

    if "stability_scores" in data.files:
        stability_scores = data["stability_scores"].astype(
            np.float32
        )
    else:
        stability_scores = np.zeros(
            mask_count,
            dtype=np.float32,
        )

    return {
        "masks": masks,
        "areas": areas,
        "bboxes": bboxes,
        "predicted_ious": predicted_ious,
        "stability_scores": stability_scores,
    }


def align_depth_to_masks(
    depth: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Resize continuous depth values to the SAM mask grid."""
    if masks.ndim != 3:
        raise ValueError(
            f"Expected masks shaped (N, H, W), got {masks.shape}"
        )

    mask_height, mask_width = masks.shape[1:]

    aligned = cv2.resize(
        depth.astype(np.float32),
        (mask_width, mask_height),
        interpolation=cv2.INTER_LINEAR,
    )

    return aligned.astype(np.float32)


def _intersection_area(
    first_mask: np.ndarray,
    first_bbox: np.ndarray,
    second_mask: np.ndarray,
    second_bbox: np.ndarray,
) -> int:
    first_x, first_y, first_w, first_h = (
        first_bbox.astype(int)
    )
    second_x, second_y, second_w, second_h = (
        second_bbox.astype(int)
    )

    x_start = max(first_x, second_x)
    y_start = max(first_y, second_y)
    x_end = min(first_x + first_w, second_x + second_w)
    y_end = min(first_y + first_h, second_y + second_h)

    if x_end <= x_start or y_end <= y_start:
        return 0

    overlap = (
        first_mask[y_start:y_end, x_start:x_end]
        & second_mask[y_start:y_end, x_start:x_end]
    )

    return int(np.count_nonzero(overlap))


def remove_near_duplicates(
    masks: np.ndarray,
    areas: np.ndarray,
    bboxes: np.ndarray,
    predicted_ious: np.ndarray,
    stability_scores: np.ndarray,
    config: MaskAnalysisConfig,
) -> tuple[list[int], dict[int, int]]:
    """
    Remove only near-identical SAM proposals.

    Smaller nested regions such as roof pieces are retained unless they
    are nearly the same size and shape as another proposal.
    """
    quality = (
        0.55 * predicted_ious
        + 0.45 * stability_scores
    )

    order = sorted(
        range(len(masks)),
        key=lambda index: (
            float(quality[index]),
            int(areas[index]),
        ),
        reverse=True,
    )

    kept: list[int] = []
    duplicate_of: dict[int, int] = {}

    for index in order:
        current_area = int(areas[index])

        if current_area < config.min_area_px:
            continue

        is_duplicate = False

        for kept_index in kept:
            kept_area = int(areas[kept_index])

            intersection = _intersection_area(
                masks[index],
                bboxes[index],
                masks[kept_index],
                bboxes[kept_index],
            )

            if intersection == 0:
                continue

            union = current_area + kept_area - intersection
            iou = intersection / union

            smaller_area = min(current_area, kept_area)
            larger_area = max(current_area, kept_area)

            containment = intersection / smaller_area
            area_ratio = smaller_area / larger_area

            duplicate = (
                iou >= config.duplicate_iou
                or (
                    containment
                    >= config.duplicate_containment
                    and area_ratio
                    >= config.duplicate_area_ratio
                )
            )

            if duplicate:
                duplicate_of[index] = kept_index
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(index)

    return kept, duplicate_of


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    minimum = float(depth.min())
    maximum = float(depth.max())

    if maximum <= minimum:
        return np.zeros_like(depth, dtype=np.float32)

    return (
        (depth - minimum) / (maximum - minimum)
    ).astype(np.float32)


def _render_kept_overlay(
    image: np.ndarray,
    masks: np.ndarray,
    kept_indices: list[int],
    median_depths: np.ndarray,
    centroids: list[tuple[float, float]],
    alpha: float,
) -> np.ndarray:
    output = image.astype(np.float32).copy()

    far_to_near = sorted(
        kept_indices,
        key=lambda index: float(median_depths[index]),
    )

    for index in far_to_near:
        depth_value = float(
            np.clip(median_depths[index], 0.0, 1.0)
        )

        color_bgr = cv2.applyColorMap(
            np.asarray([[round(depth_value * 255)]], dtype=np.uint8),
            cv2.COLORMAP_TURBO,
        )[0, 0]

        color_rgb = color_bgr[::-1].astype(np.float32)
        mask = masks[index]

        output[mask] = (
            output[mask] * (1.0 - alpha)
            + color_rgb * alpha
        )

    output = np.clip(output, 0, 255).astype(np.uint8)

    for index in far_to_near:
        centroid_x, centroid_y = centroids[index]

        cv2.putText(
            output,
            str(index),
            (int(centroid_x), int(centroid_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output


def run_mask_analysis(
    masks_path: Path,
    depth_path: Path,
    image_path: Path,
    output_dir: Path,
    config: MaskAnalysisConfig | None = None,
) -> dict[str, int]:
    config = config or MaskAnalysisConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_mask_candidates(masks_path)

    masks = candidates["masks"]
    areas = candidates["areas"]
    bboxes = candidates["bboxes"]
    predicted_ious = candidates["predicted_ious"]
    stability_scores = candidates["stability_scores"]

    depth = np.load(depth_path).astype(np.float32)
    aligned_depth = align_depth_to_masks(depth, masks)
    normalized_depth = _normalize_depth(aligned_depth)

    np.save(
        output_dir / "depth_aligned.npy",
        aligned_depth,
    )

    cv2.imwrite(
        str(output_dir / "depth_aligned_preview.png"),
        np.round(normalized_depth * 255).astype(np.uint8),
    )

    kept_indices, duplicate_of = remove_near_duplicates(
        masks=masks,
        areas=areas,
        bboxes=bboxes,
        predicted_ious=predicted_ious,
        stability_scores=stability_scores,
        config=config,
    )

    median_depths = np.zeros(len(masks), dtype=np.float32)
    centroids: list[tuple[float, float]] = []
    analysis: list[dict] = []

    image_area = masks.shape[1] * masks.shape[2]

    for index, mask in enumerate(masks):
        y_values, x_values = np.nonzero(mask)

        if len(x_values) == 0:
            centroid = (0.0, 0.0)
            values = np.asarray([0.0], dtype=np.float32)
        else:
            centroid = (
                float(x_values.mean()),
                float(y_values.mean()),
            )
            values = normalized_depth[mask]

        centroids.append(centroid)

        median_depth = float(np.median(values))
        median_depths[index] = median_depth

        analysis.append(
            {
                "id": index,
                "area": int(areas[index]),
                "coverage": float(areas[index] / image_area),
                "bbox_xywh": [
                    float(value)
                    for value in bboxes[index]
                ],
                "predicted_iou": float(
                    predicted_ious[index]
                ),
                "stability_score": float(
                    stability_scores[index]
                ),
                "median_depth": median_depth,
                "mean_depth": float(np.mean(values)),
                "depth_p10": float(np.percentile(values, 10)),
                "depth_p90": float(np.percentile(values, 90)),
                "depth_spread": float(
                    np.percentile(values, 90)
                    - np.percentile(values, 10)
                ),
                "centroid_xy": [
                    centroid[0],
                    centroid[1],
                ],
                "kept": index in kept_indices,
                "duplicate_of": duplicate_of.get(index),
            }
        )

    near_first = sorted(
        kept_indices,
        key=lambda index: float(median_depths[index]),
        reverse=True,
    )

    near_rank = {
        mask_index: rank
        for rank, mask_index in enumerate(near_first)
    }

    for item in analysis:
        item["near_rank"] = near_rank.get(item["id"])

    with open(
        output_dir / "mask_analysis.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(analysis, file, indent=2)

    with open(
        output_dir / "kept_indices.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(near_first, file, indent=2)

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    mask_height, mask_width = masks.shape[1:]

    image = cv2.resize(
        image,
        (mask_width, mask_height),
        interpolation=cv2.INTER_AREA,
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overlay = _render_kept_overlay(
        image=image,
        masks=masks,
        kept_indices=kept_indices,
        median_depths=median_depths,
        centroids=centroids,
        alpha=config.overlay_alpha,
    )

    cv2.imwrite(
        str(output_dir / "kept_overlay.png"),
        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
    )

    return {
        "input_masks": int(len(masks)),
        "kept_masks": int(len(kept_indices)),
        "duplicates_removed": int(len(duplicate_of)),
    }
