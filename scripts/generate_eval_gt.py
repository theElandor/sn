import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one random-rotation GT JSON file.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"))
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=None,
        help="Optional folder with train.txt, val.txt, and optional test.txt entries.",
    )
    parser.add_argument("--gt-json", type=Path, default=Path("data/gt/ground_truth.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    gt_json = args.gt_json.expanduser().resolve()
    candidates = discover_stl_files(input_dir)
    scans = collect_fold_scans(args.fold_dir, candidates) if args.fold_dir else candidates

    rng = np.random.default_rng(args.seed)
    rotations = []
    for scan in scans:
        angles = rng.uniform(0.0, 2.0 * math.pi, size=3)
        matrix = euler_xyz_matrix(angles)
        rotations.append(
            {
                "scan": scan.relative_to(input_dir).as_posix(),
                "angles_rad": angles.tolist(),
                "rotation_matrix": matrix.tolist(),
            }
        )

    gt_json.parent.mkdir(parents=True, exist_ok=True)
    gt_json.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "input_dir": str(input_dir),
                "rotations": rotations,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {len(scans)} GT rotations to {gt_json}")


def discover_stl_files(root):
    root = Path(root).expanduser()
    files = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".stl"
    )
    if not files:
        raise RuntimeError(f"No STL files found under {root}")
    return files


def collect_fold_scans(fold_dir, candidates):
    fold_dir = Path(fold_dir).expanduser()
    split_paths = [fold_dir / "train.txt", fold_dir / "val.txt"]
    test_split = fold_dir / "test.txt"
    if test_split.exists():
        split_paths.append(test_split)

    scans = []
    for split_path in split_paths:
        scans.extend(read_split_files(split_path, candidates))

    unique_scans = sorted(set(scans))
    if not unique_scans:
        raise RuntimeError(f"No scans matched split files under {fold_dir}")
    return unique_scans


def read_split_files(split_path, candidates):
    if not split_path.exists():
        raise RuntimeError(f"Missing split file: {split_path}")

    files = []
    candidate_text = [path.as_posix().casefold() for path in candidates]
    for line_number, line in enumerate(split_path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        identifier, arch = parse_split_entry(split_path, line_number, line)
        if arch is None:
            files.extend(
                match_patient_scans(
                    split_path,
                    line_number,
                    identifier,
                    candidates,
                    candidate_text,
                )
            )
        else:
            files.append(
                match_scan_tuple(
                    split_path,
                    line_number,
                    identifier,
                    arch,
                    candidates,
                    candidate_text,
                )
            )

    if not files:
        raise RuntimeError(f"Split file is empty: {split_path}")
    return files


def parse_split_entry(split_path, line_number, line):
    parts = line.split()
    if len(parts) not in (1, 2):
        raise RuntimeError(
            f"Invalid split entry at {split_path}:{line_number}: expected "
            f"'<identifier>' or '<identifier> <arch>', got '{line}'"
        )
    if any(Path(part).suffix for part in parts):
        raise RuntimeError(
            f"Invalid split entry at {split_path}:{line_number}: split files must use "
            f"'<identifier>' or '<identifier> <arch>' entries without mesh "
            f"extensions, got '{line}'"
        )
    return parts[0], parts[1] if len(parts) == 2 else None


def match_scan_tuple(
    split_path,
    line_number,
    identifier,
    arch,
    candidates,
    candidate_text,
):
    tokens = (identifier.casefold(), arch.casefold())
    matches = [
        path
        for path, text in zip(candidates, candidate_text)
        if all(token in text for token in tokens)
    ]
    if not matches:
        raise RuntimeError(
            f"No STL file matched split tuple '{identifier} {arch}' "
            f"from {split_path}:{line_number}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Split tuple '{identifier} {arch}' from {split_path}:{line_number} "
            f"matched multiple STL files: {format_matches(matches)}"
        )
    return matches[0]


def match_patient_scans(split_path, line_number, identifier, candidates, candidate_text):
    token = identifier.casefold()
    matches = [
        path
        for path, text in zip(candidates, candidate_text)
        if token in text and has_lower_or_upper_name(path)
    ]
    if not matches:
        raise RuntimeError(
            f"No lower/upper STL files matched patient identifier '{identifier}' "
            f"from {split_path}:{line_number}"
        )
    return matches


def has_lower_or_upper_name(path):
    name = path.name.casefold()
    return "lower" in name or "upper" in name


def format_matches(matches):
    formatted = ", ".join(str(path) for path in matches[:5])
    if len(matches) > 5:
        formatted += ", ..."
    return formatted


def euler_xyz_matrix(angles):
    x, y, z = angles
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)

    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz @ ry @ rx


if __name__ == "__main__":
    main()
