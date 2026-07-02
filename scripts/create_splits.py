import argparse
from pathlib import Path
import random


SEED = 42
N_FOLDS = 5
VAL_RATIO = 0.05
TEST_RATIO = 0.05


def parse_args():
    parser = argparse.ArgumentParser(description="Create train/val/test split files for scans.")
    parser.add_argument("--data-root", type=Path, default=Path("data/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser()

    samples = sorted(path.name for path in data_root.glob("*.stl"))
    if not samples:
        raise RuntimeError(f"No STL files found in {data_root}")

    n_val = max(1, round(len(samples) * args.val_ratio))
    n_test = max(1, round(len(samples) * args.test_ratio))

    for fold in range(args.folds):
        fold_samples = samples[:]
        random.Random(args.seed + fold).shuffle(fold_samples)

        val = sorted(fold_samples[:n_val])
        test = sorted(fold_samples[n_val : n_val + n_test])
        train = sorted(fold_samples[n_val + n_test :])

        fold_dir = output_dir / f"fold_{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        write_lines(fold_dir / "train.txt", train)
        write_lines(fold_dir / "val.txt", val)
        write_lines(fold_dir / "test.txt", test)

        print(
            f"{fold_dir.name}: train={len(train)} val={len(val)} test={len(test)}"
        )


def write_lines(path, lines):
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
