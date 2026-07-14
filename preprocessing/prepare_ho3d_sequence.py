#!/usr/bin/env python3
"""Convert one HO3D sequence into the MV-SAM3D multi-view input layout.

The selector prioritizes frames with a large visible object mask while enforcing
a minimum temporal gap between selected frames.  Source masks are expected to
use the ``obj_mask_white`` convention: white object, black background.

Example:
    python preprocessing/prepare_ho3d_sequence.py \
        --seq ABF10 \
        --interval 40 \
        --num-frames 8
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HO3D_ROOT = Path("/home/mengxiangting/nas/mengxt/Datasets/HO3D_v3/train")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
MASK_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class FrameRecord:
    frame_name: str
    frame_number: int
    image_path: Path
    mask_path: Path
    mask_pixels: int
    mask_ratio: float


def natural_key(path: Path) -> tuple[int, int | str]:
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.stem)


def find_matching_file(directory: Path, stem: str, suffixes: Iterable[str]) -> Path | None:
    for candidate_stem in (stem, f"{stem}_mask"):
        for suffix in suffixes:
            candidate = directory / f"{candidate_stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def resolve_sequence(seq_arg: str, ho3d_root: Path) -> Path:
    direct = Path(seq_arg).expanduser()
    seq_dir = direct if direct.is_dir() else ho3d_root.expanduser() / seq_arg
    seq_dir = seq_dir.resolve()
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"HO3D sequence not found: {seq_dir}")
    return seq_dir


def infer_object_name(seq_dir: Path, frame_names: Iterable[str]) -> str:
    """Read and validate the HO3D ``objName`` label for a sequence."""
    meta_dir = seq_dir / "meta"
    if not meta_dir.is_dir():
        raise FileNotFoundError(
            f"HO3D metadata directory not found: {meta_dir}. "
            "Pass --mask-prompt explicitly only if this is a custom sequence."
        )

    object_names: set[str] = set()
    missing_frames: list[str] = []
    for frame_name in frame_names:
        meta_path = meta_dir / f"{frame_name}.pkl"
        if not meta_path.is_file():
            missing_frames.append(frame_name)
            continue
        with open(meta_path, "rb") as handle:
            metadata = pickle.load(handle, encoding="latin1")
        raw_name = metadata.get("objName") if isinstance(metadata, dict) else None
        if raw_name is None:
            missing_frames.append(frame_name)
            continue
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8")
        object_name = str(raw_name).strip()
        if object_name:
            object_names.add(object_name)

    if not object_names:
        missing = ",".join(missing_frames[:5])
        raise RuntimeError(
            f"Could not read objName from selected HO3D metadata (examples: {missing}). "
            "Use --mask-prompt <YCB_NAME> to override."
        )
    if len(object_names) != 1:
        raise RuntimeError(
            f"Selected frames contain multiple HO3D objName labels: {sorted(object_names)}"
        )

    object_name = next(iter(object_names))
    if object_name in {".", ".."} or Path(object_name).name != object_name:
        raise ValueError(f"Unsafe HO3D objName value: {object_name!r}")
    return object_name


def collect_records(
    seq_dir: Path,
    mask_dir_name: str,
    start_frame: int | None,
    end_frame: int | None,
    min_mask_ratio: float,
    max_mask_ratio: float,
) -> tuple[list[FrameRecord], list[str]]:
    rgb_dir = seq_dir / "rgb"
    mask_dir = seq_dir / mask_dir_name
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Object-mask directory not found: {mask_dir}")

    image_paths = sorted(
        (path for path in rgb_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    records: list[FrameRecord] = []
    skipped: list[str] = []

    for ordinal, image_path in enumerate(image_paths):
        try:
            frame_number = int(image_path.stem)
        except ValueError:
            frame_number = ordinal

        if start_frame is not None and frame_number < start_frame:
            continue
        if end_frame is not None and frame_number > end_frame:
            continue

        mask_path = find_matching_file(mask_dir, image_path.stem, MASK_SUFFIXES)
        if mask_path is None:
            skipped.append(f"{image_path.stem}: missing mask")
            continue

        with Image.open(image_path) as image, Image.open(mask_path) as mask_image:
            image_size = image.size
            mask_gray = np.asarray(mask_image.convert("L"))

        if mask_gray.shape != (image_size[1], image_size[0]):
            skipped.append(
                f"{image_path.stem}: size mismatch image={image_size[::-1]} "
                f"mask={mask_gray.shape}"
            )
            continue

        object_mask = mask_gray > 127
        mask_pixels = int(object_mask.sum())
        mask_ratio = float(object_mask.mean())
        if mask_pixels == 0:
            skipped.append(f"{image_path.stem}: empty mask")
            continue
        if mask_ratio < min_mask_ratio or mask_ratio > max_mask_ratio:
            skipped.append(
                f"{image_path.stem}: mask ratio {mask_ratio:.4f} outside "
                f"[{min_mask_ratio:.4f}, {max_mask_ratio:.4f}]"
            )
            continue

        records.append(
            FrameRecord(
                frame_name=image_path.stem,
                frame_number=frame_number,
                image_path=image_path.resolve(),
                mask_path=mask_path.resolve(),
                mask_pixels=mask_pixels,
                mask_ratio=mask_ratio,
            )
        )

    return records, skipped


def select_records(records: list[FrameRecord], num_frames: int, interval: int) -> list[FrameRecord]:
    """Greedily maximize visible mask area subject to a minimum frame gap."""
    if num_frames <= 0:
        raise ValueError("--num-frames must be greater than zero")
    if interval < 0:
        raise ValueError("--interval must be zero or greater")

    ranked = sorted(records, key=lambda item: (-item.mask_pixels, item.frame_number))
    selected: list[FrameRecord] = []
    for candidate in ranked:
        if all(
            abs(candidate.frame_number - existing.frame_number) >= interval
            for existing in selected
        ):
            selected.append(candidate)
            if len(selected) == num_frames:
                break
    return sorted(selected, key=lambda item: item.frame_number)


def write_dataset(
    selected: list[FrameRecord],
    output_dir: Path,
    mask_prompt: str,
    overwrite: bool,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}. "
                "Use --overwrite to replace this generated dataset."
            )
        shutil.rmtree(output_dir)

    images_dir = output_dir / "images"
    masks_dir = output_dir / mask_prompt
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for record in selected:
        with Image.open(record.image_path) as source_image:
            rgb = np.asarray(source_image.convert("RGB"))
        with Image.open(record.mask_path) as source_mask:
            object_mask = np.asarray(source_mask.convert("L")) > 127

        rgba = np.zeros((*object_mask.shape, 4), dtype=np.uint8)
        rgba[object_mask, :3] = rgb[object_mask]
        rgba[object_mask, 3] = 255

        Image.fromarray(rgb, mode="RGB").save(images_dir / f"{record.frame_name}.png")
        Image.fromarray(rgba, mode="RGBA").save(masks_dir / f"{record.frame_name}.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one HO3D sequence to data/ho3d/<seq>/, selecting large "
            "obj_mask_white frames with a configurable temporal gap."
        )
    )
    parser.add_argument(
        "--seq",
        required=True,
        help="HO3D sequence name (for example ABF10) or its full directory path.",
    )
    parser.add_argument(
        "--ho3d-root",
        type=Path,
        default=DEFAULT_HO3D_ROOT,
        help=f"HO3D split root (default: {DEFAULT_HO3D_ROOT}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "ho3d",
        help="Output parent; the sequence name is appended automatically.",
    )
    parser.add_argument("--mask-dir", default="obj_mask_white")
    parser.add_argument(
        "--mask-prompt",
        default=None,
        help="Mask folder name. By default it is inferred from meta/<frame>.pkl objName.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Maximum number of frames to select (default: 8).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Minimum difference between selected frame numbers (default: 30).",
    )
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument(
        "--min-mask-ratio",
        type=float,
        default=0.0,
        help="Discard masks smaller than this image-area ratio.",
    )
    parser.add_argument(
        "--max-mask-ratio",
        type=float,
        default=1.0,
        help="Discard masks larger than this image-area ratio.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected frames without writing files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated sequence directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_mask_ratio <= args.max_mask_ratio <= 1.0:
        raise ValueError("Mask ratios must satisfy 0 <= min <= max <= 1")

    seq_dir = resolve_sequence(args.seq, args.ho3d_root)
    output_dir = args.output_root.expanduser().resolve() / seq_dir.name
    records, skipped = collect_records(
        seq_dir=seq_dir,
        mask_dir_name=args.mask_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        min_mask_ratio=args.min_mask_ratio,
        max_mask_ratio=args.max_mask_ratio,
    )
    selected = select_records(records, args.num_frames, args.interval)
    if not selected:
        raise RuntimeError("No usable RGB/object-mask pairs were selected")
    mask_prompt = args.mask_prompt or infer_object_name(
        seq_dir, (record.frame_name for record in selected)
    )

    print(f"Sequence: {seq_dir}")
    print(f"YCB object: {mask_prompt}")
    print(f"Usable frames: {len(records)}; skipped: {len(skipped)}")
    print(
        f"Selected {len(selected)}/{args.num_frames} frames "
        f"with minimum interval {args.interval}:"
    )
    for record in selected:
        print(
            f"  {record.frame_name}: mask_pixels={record.mask_pixels}, "
            f"mask_ratio={record.mask_ratio:.2%}"
        )
    if len(selected) < args.num_frames:
        print(
            "Warning: fewer frames were selected than requested; reduce --interval "
            "or relax the mask-ratio filters."
        )

    if args.dry_run:
        print("Dry run: no files written.")
        return

    write_dataset(selected, output_dir, mask_prompt, args.overwrite)
    manifest = {
        "source_sequence": str(seq_dir),
        "output_directory": str(output_dir),
        "mask_prompt": mask_prompt,
        "selection": {
            "num_frames_requested": args.num_frames,
            "interval": args.interval,
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "min_mask_ratio": args.min_mask_ratio,
            "max_mask_ratio": args.max_mask_ratio,
        },
        "selected_frames": [
            {
                **asdict(record),
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path),
            }
            for record in selected
        ],
        "skipped_frames": skipped,
    }
    with open(output_dir / "selection.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    image_names = ",".join(record.frame_name for record in selected)
    print(f"Dataset written to: {output_dir}")
    print("Next MV-SAM3D arguments:")
    print(f"  --input_path {output_dir}")
    print(f"  --mask_prompt {mask_prompt}")
    print(f"  --image_names {image_names}")


if __name__ == "__main__":
    main()
