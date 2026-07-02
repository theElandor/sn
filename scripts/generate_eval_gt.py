import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Generate one random-rotation GT JSON file.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"))
    parser.add_argument("--gt-json", type=Path, default=Path("data/gt/ground_truth.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    gt_json = args.gt_json.expanduser().resolve()
    scans = sorted(path for path in input_dir.rglob("*.stl") if path.is_file())
    if not scans:
        raise RuntimeError(f"No STL files found under {input_dir}")

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
