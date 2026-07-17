#!/usr/bin/env python3
"""Project an MV-SAM3D result back onto its selected HO3D input views.

The pose stored in params.npz maps the canonical mesh into PyTorch3D camera
coordinates of MV view 0.  View 0 is therefore the only pose-free scale anchor.
Other views can be rendered through DA3 extrinsics for diagnostics, but they
assume a static object and are not used for automatic scale selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh
from PIL import Image
from pytorch3d.transforms import Transform3d, quaternion_to_matrix


Z_UP_TO_Y_UP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--da3_output", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--scale_factors",
        default="0.75,0.80,0.85,0.90,0.95,1.00,1.05,1.10,1.15,1.20,1.25",
        help="Comma-separated corrections multiplying params.npz scale.",
    )
    parser.add_argument(
        "--diagnostic_all_views",
        action="store_true",
        help="Also propagate view-0 pose with DA3 extrinsics (diagnostic only).",
    )
    parser.add_argument("--surface_samples", type=int, default=200000)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def as_4x4(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        output = np.eye(4, dtype=np.float64)
        output[:3] = matrix
        return output
    raise ValueError(f"Expected a 3x4 or 4x4 matrix, got {matrix.shape}")


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    meshes = [item for item in loaded.dump() if isinstance(item, trimesh.Trimesh)]
    if not meshes:
        raise RuntimeError(f"No triangle mesh found in {path}")
    return trimesh.util.concatenate(meshes)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(Image.open(path))
    if image.ndim == 3 and image.shape[2] == 4:
        mask = image[..., 3] > 0
    elif image.ndim == 3:
        mask = np.any(image[..., :3] > 0, axis=2)
    else:
        mask = image > 0
    if mask.shape != shape:
        mask = cv2.resize(mask.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    return mask


def camera_intrinsics(matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    # DA3 versions may emit normalized intrinsics instead of pixel intrinsics.
    if matrix[0, 0] < 10.0 and matrix[1, 1] < 10.0:
        matrix[0, :] *= width
        matrix[1, :] *= height
    return matrix


def transform_view0_points(
    canonical_points: np.ndarray,
    params: dict[str, np.ndarray],
    correction: float,
) -> np.ndarray:
    scale = np.asarray(params["scale"], dtype=np.float32).reshape(-1)
    rotation = np.asarray(params["rotation"], dtype=np.float32).reshape(-1)[:4]
    translation = np.asarray(params["translation"], dtype=np.float32).reshape(-1)[:3]
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    scale = scale[:3] * float(correction)

    transform = (
        Transform3d(dtype=torch.float32)
        .scale(torch.from_numpy(scale).reshape(1, 3))
        .rotate(quaternion_to_matrix(torch.from_numpy(rotation).reshape(1, 4)))
        .translate(torch.from_numpy(translation).reshape(1, 3))
    )
    points = canonical_points.astype(np.float32) @ Z_UP_TO_Y_UP.T
    points_p3d = transform.transform_points(torch.from_numpy(points)[None])[0].numpy()
    return points_p3d.astype(np.float64) @ P3D_TO_CV.T


def move_view0_to_view(
    points_view0: np.ndarray,
    extrinsics: np.ndarray,
    view_index: int,
) -> np.ndarray:
    if view_index == 0:
        return points_view0
    view0_to_world = np.linalg.inv(as_4x4(extrinsics[0]))
    world_to_view = as_4x4(extrinsics[view_index])
    transform = world_to_view @ view0_to_world
    points_h = np.concatenate([points_view0, np.ones((len(points_view0), 1))], axis=1)
    return (points_h @ transform.T)[:, :3]


def rasterize_points(
    points: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-5)
    points = points[valid]
    projected = points @ intrinsics.T
    pixels = projected[:, :2] / projected[:, 2:3]
    pixels = np.rint(pixels).astype(np.int32)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels = pixels[inside]
    silhouette = np.zeros((height, width), dtype=np.uint8)
    if len(pixels):
        silhouette[pixels[:, 1], pixels[:, 0]] = 1
        kernel_size = max(1, 2 * int(radius) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        silhouette = cv2.dilate(silhouette, kernel)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, kernel)
    return silhouette.astype(bool), pixels


def mask_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    intersection = np.count_nonzero(prediction & target)
    union = np.count_nonzero(prediction | target)
    pred_area = np.count_nonzero(prediction)
    target_area = np.count_nonzero(target)
    return {
        "iou": float(intersection / max(1, union)),
        "mask_recall": float(intersection / max(1, target_area)),
        "projection_precision": float(intersection / max(1, pred_area)),
        "projected_to_mask_area": float(pred_area / max(1, target_area)),
    }


def draw_overlay(
    rgb: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    label: str,
) -> np.ndarray:
    output = rgb.copy()
    target_contours, _ = cv2.findContours(target.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_contours, _ = cv2.findContours(prediction.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = output.copy()
    overlay[prediction] = (255, 0, 190)
    output = cv2.addWeighted(output, 0.72, overlay, 0.28, 0.0)
    cv2.drawContours(output, target_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(output, pred_contours, -1, (255, 0, 190), 2)
    cv2.putText(output, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(output, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def main() -> None:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    da3_output = args.da3_output.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads((input_path / "selection.json").read_text())
    frames = [str(row["frame_name"]) for row in selection["selected_frames"]]
    mask_prompt = str(selection["mask_prompt"])
    da3 = np.load(da3_output, allow_pickle=True)
    params_npz = np.load(result_dir / "params.npz", allow_pickle=True)
    params = {key: params_npz[key] for key in params_npz.files}
    required = {"scale", "rotation", "translation"}
    missing = required - set(params)
    if missing:
        raise KeyError(f"Missing MV-SAM3D pose fields in params.npz: {sorted(missing)}")

    da3_names = [Path(str(path)).stem for path in da3["image_files"]]
    index_by_name = {name: index for index, name in enumerate(da3_names)}
    order = [index_by_name[name] for name in frames]
    extrinsics = np.asarray(da3["extrinsics"])[order]
    intrinsics = np.asarray(da3["intrinsics"])[order]

    mesh = load_mesh(result_dir / "result.glb")
    np.random.seed(args.seed)
    count = min(max(1000, args.surface_samples), max(args.surface_samples, len(mesh.vertices)))
    surface_points, _ = trimesh.sample.sample_surface(mesh, count=count)
    scale_factors = [float(value) for value in args.scale_factors.split(",") if value.strip()]
    if not scale_factors:
        raise ValueError("--scale_factors must contain at least one value")

    view_indices = list(range(len(frames))) if args.diagnostic_all_views else [0]
    rows = []
    best_view0 = None
    for correction in scale_factors:
        points_view0 = transform_view0_points(surface_points, params, correction)
        for view_index in view_indices:
            frame = frames[view_index]
            rgb_path = input_path / "images" / f"{frame}.png"
            mask_path = input_path / mask_prompt / f"{frame}.png"
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            height, width = rgb.shape[:2]
            target = load_mask(mask_path, (height, width))
            points = move_view0_to_view(points_view0, extrinsics, view_index)
            matrix = camera_intrinsics(intrinsics[view_index], width, height)
            prediction, _ = rasterize_points(points, matrix, width, height, args.point_radius)
            metrics = mask_metrics(prediction, target)
            row = {
                "frame": frame,
                "view_index": view_index,
                "scale_correction": correction,
                **metrics,
            }
            rows.append(row)
            if view_index == 0 and (best_view0 is None or row["iou"] > best_view0["iou"]):
                best_view0 = row
            label = (
                f"view={view_index} frame={frame} correction={correction:.3f} "
                f"IoU={metrics['iou']:.3f} area={metrics['projected_to_mask_area']:.3f}"
            )
            output = draw_overlay(rgb, target, prediction, label)
            cv2.imwrite(
                str(out_dir / f"view{view_index:02d}_{frame}_scale_{correction:.3f}.jpg"),
                cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
            )

    raw_scale = np.asarray(params["scale"], dtype=np.float64).reshape(-1).tolist()
    summary = {
        "source": "mvsam3d_params_view0_pose_projection",
        "input_path": str(input_path),
        "result_dir": str(result_dir),
        "da3_output": str(da3_output),
        "mask_prompt": mask_prompt,
        "selected_frames": frames,
        "raw_params_scale": raw_scale,
        "view0_is_scale_anchor": True,
        "other_views_are_static_object_diagnostics_only": True,
        "recommended_scale_correction": best_view0["scale_correction"],
        "recommended_final_scale": (np.asarray(raw_scale) * best_view0["scale_correction"]).tolist(),
        "best_view0": best_view0,
        "rows": rows,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in [
        "raw_params_scale",
        "recommended_scale_correction",
        "recommended_final_scale",
        "best_view0",
    ]}, indent=2))
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
