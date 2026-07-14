#!/usr/bin/env python3
"""Run the approved HO3D sequences through prepare, DA3, and MV-SAM3D."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPROVED_SEQUENCES = (
    "ABF10", "ABF11", "ABF12", "ABF13", "ABF14",
    "GPMF12", "GPMF13", "GPMF14",
    "MC1", "MC2", "MC4", "MC6",
    "ND2",
    "SB10", "SB12", "SB14",
    "SM2", "SM3", "SM4", "SM5",
    "SMu1", "SMu40", "SMu41",
    "SS1", "SS2",
    "ShSu10", "ShSu12", "ShSu14",
    "SiS1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable prepare -> DA3 -> MV-SAM3D batch for approved HO3D sequences."
    )
    parser.add_argument("--sequences", nargs="+", default=list(APPROVED_SEQUENCES))
    parser.add_argument(
        "--ho3d_root",
        default="/home/mengxiangting/nas/mengxt/Datasets/HO3D_v3/train",
    )
    parser.add_argument("--data_root", default=str(PROJECT_ROOT / "data" / "ho3d"))
    parser.add_argument("--da3_output_root", default=str(PROJECT_ROOT / "da3_outputs" / "ho3d"))
    parser.add_argument(
        "--da3_model_path",
        default="/home/mengxiangting/nas/mengxt/Projects/Depth-Anything-3/DA3_GAINT",
    )
    parser.add_argument(
        "--da3_python",
        default="/home/mengxiangting/nas/mengxt/anaconda3/envs/da3/bin/python",
    )
    parser.add_argument(
        "--sam3d_python",
        default="/home/mengxiangting/nas/mengxt/anaconda3/envs/sam3d-objects/bin/python",
    )
    parser.add_argument(
        "--run_root",
        default=str(PROJECT_ROOT / "batch_runs" / "ho3d_approved_mv_v1"),
    )
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--interval", type=int, default=50)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--min_mask_ratio", type=float, default=0.01)
    parser.add_argument("--max_mask_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1_steps", type=int, default=50)
    parser.add_argument("--stage2_steps", type=int, default=25)
    parser.add_argument("--stage1_entropy_alpha", type=float, default=30.0)
    parser.add_argument("--stage2_entropy_alpha", type=float, default=30.0)
    parser.add_argument("--decode_formats", default="gaussian,mesh")
    parser.add_argument("--force_prepare", action="store_true")
    parser.add_argument("--force_da3", action="store_true")
    parser.add_argument("--force_sam3d", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def environment_for(python_path: Path, cuda_visible_devices: str) -> dict[str, str]:
    prefix = python_path.resolve().parents[1]
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    env["CONDA_PREFIX"] = str(prefix)
    env["CONDA_DEFAULT_ENV"] = prefix.name
    env["PATH"] = f"{prefix / 'bin'}:{env.get('PATH', '')}"
    library_paths = [
        prefix / "lib",
        prefix / "x86_64-conda-linux-gnu" / "sysroot" / "usr" / "lib64",
        prefix / "lib" / "python3.11" / "site-packages" / "open3d",
        Path("/usr/local/cuda/lib64"),
        Path("/usr/local/nvidia/lib"),
        Path("/usr/local/nvidia/lib64"),
    ]
    env["LD_LIBRARY_PATH"] = ":".join(str(path) for path in library_paths if path.exists())
    return env


def run_logged(
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("  command:", " ".join(command), flush=True)
    print("  log:", log_path, flush=True)
    if dry_run:
        return
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND\n" + " ".join(command) + "\n\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def valid_sam3d_output(path: str | Path | None) -> bool:
    if not path:
        return False
    directory = Path(path)
    return (
        (directory / "result.glb").is_file()
        and (directory / "params.npz").is_file()
    )


def newest_result(sequence: str, prompt: str, started_at: float) -> Path | None:
    root = PROJECT_ROOT / "visualization" / sequence / prompt
    candidates = sorted(
        root.glob("*/result.glb") if root.is_dir() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if path.stat().st_mtime >= started_at - 2.0 and (path.parent / "params.npz").is_file():
            return path.parent.resolve()
    return None


def main() -> None:
    args = parse_args()
    sequences = list(dict.fromkeys(str(value) for value in args.sequences))
    unknown = sorted(set(sequences) - set(APPROVED_SEQUENCES))
    if unknown:
        raise ValueError(f"Sequences are not in the approved list: {unknown}")

    da3_python = Path(args.da3_python).expanduser().resolve()
    sam3d_python = Path(args.sam3d_python).expanduser().resolve()
    for path in (da3_python, sam3d_python):
        if not path.is_file():
            raise FileNotFoundError(path)

    run_root = Path(args.run_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    da3_root = Path(args.da3_output_root).expanduser().resolve()
    status_path = run_root / "status.json"
    status = load_json(status_path, {"source": "ho3d_approved_mv_batch", "jobs": {}})
    status["approved_sequences"] = list(APPROVED_SEQUENCES)
    status["requested_sequences"] = sequences
    status["settings"] = {
        key: value for key, value in vars(args).items() if key not in {"sequences"}
    }

    env_da3 = environment_for(da3_python, args.cuda_visible_devices)
    env_sam3d = environment_for(sam3d_python, args.cuda_visible_devices)
    failures = 0

    for index, sequence in enumerate(sequences, start=1):
        print(f"\n[{index}/{len(sequences)}] {sequence}", flush=True)
        job = status["jobs"].setdefault(sequence, {"sequence": sequence})
        sequence_data = data_root / sequence
        selection_path = sequence_data / "selection.json"
        da3_dir = da3_root / sequence
        da3_npz = da3_dir / "da3_output.npz"
        log_dir = run_root / "logs" / sequence

        try:
            prepare_valid = selection_path.is_file()
            rerun_prepare = bool(args.force_prepare or not prepare_valid)
            if rerun_prepare:
                command = [
                    str(sam3d_python),
                    str(PROJECT_ROOT / "preprocessing" / "prepare_ho3d_sequence.py"),
                    "--seq", sequence,
                    "--ho3d-root", str(Path(args.ho3d_root).expanduser()),
                    "--output-root", str(data_root),
                    "--interval", str(args.interval),
                    "--num-frames", str(args.num_frames),
                    "--min-mask-ratio", str(args.min_mask_ratio),
                    "--max-mask-ratio", str(args.max_mask_ratio),
                ]
                if args.force_prepare and sequence_data.exists():
                    command.append("--overwrite")
                run_logged(command, log_dir / "prepare.log", env_sam3d, args.dry_run)
            if not args.dry_run and not selection_path.is_file():
                raise RuntimeError(f"prepare did not write {selection_path}")
            job["prepare"] = "dry_run" if args.dry_run else "valid"
            job["selection_json"] = str(selection_path)
            write_json(status_path, status)

            rerun_da3 = bool(args.force_da3 or rerun_prepare or not da3_npz.is_file())
            if rerun_da3:
                command = [
                    str(da3_python),
                    str(PROJECT_ROOT / "scripts" / "run_da3.py"),
                    "--image_dir", str(sequence_data / "images"),
                    "--output_dir", str(da3_dir),
                    "--model_path", str(Path(args.da3_model_path).expanduser()),
                ]
                run_logged(command, log_dir / "da3.log", env_da3, args.dry_run)
            if not args.dry_run and not da3_npz.is_file():
                raise RuntimeError(f"DA3 did not write {da3_npz}")
            job["da3"] = "dry_run" if args.dry_run else "valid"
            job["da3_output"] = str(da3_npz)
            write_json(status_path, status)

            selection = load_json(selection_path, {}) if selection_path.is_file() else {}
            prompt = str(selection.get("mask_prompt", "")).strip()
            if not prompt and not args.dry_run:
                raise RuntimeError(f"mask_prompt is missing from {selection_path}")
            job["mask_prompt"] = prompt

            previous_output = job.get("sam3d_output")
            rerun_sam3d = bool(
                args.force_sam3d
                or rerun_da3
                or not valid_sam3d_output(previous_output)
            )
            if rerun_sam3d:
                command = [
                    str(sam3d_python),
                    str(PROJECT_ROOT / "run_inference_weighted.py"),
                    "--input_path", str(sequence_data),
                    "--mask_prompt", prompt or "DRY_RUN_PROMPT",
                    "--da3_output", str(da3_npz),
                    "--seed", str(args.seed),
                    "--stage1_steps", str(args.stage1_steps),
                    "--stage2_steps", str(args.stage2_steps),
                    "--stage1_entropy_alpha", str(args.stage1_entropy_alpha),
                    "--stage2_entropy_alpha", str(args.stage2_entropy_alpha),
                    "--decode_formats", str(args.decode_formats),
                ]
                started_at = time.time()
                run_logged(command, log_dir / "sam3d.log", env_sam3d, args.dry_run)
                output = None if args.dry_run else newest_result(sequence, prompt, started_at)
                if not args.dry_run and output is None:
                    raise RuntimeError("MV-SAM3D finished but no new result.glb + params.npz was found")
                job["sam3d_output"] = None if output is None else str(output)
            job["sam3d"] = "dry_run" if args.dry_run else "valid"
            job.pop("error", None)
            print("  done:", job.get("sam3d_output"), flush=True)
        except Exception as error:
            failures += 1
            job["error"] = f"{type(error).__name__}: {error}"
            print("  failed:", job["error"], flush=True)
            if args.stop_on_error:
                write_json(status_path, status)
                raise
        finally:
            write_json(status_path, status)

    valid = sum(valid_sam3d_output(row.get("sam3d_output")) for row in status["jobs"].values())
    summary = {
        "num_requested": len(sequences),
        "num_valid_sam3d": int(valid),
        "num_failed_this_run": failures,
        "status_json": str(status_path),
    }
    write_json(run_root / "summary.json", summary)
    print("\n" + json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
