import argparse
from pathlib import Path
import random
import re


SEED = 42
N_FOLDS = 5
VAL_RATIO = 0.05
TEST_RATIO = 0.05
ARCH_TOKENS = {
    "lower": "lower",
    "mandibular": "lower",
    "upper": "upper",
    "maxillary": "upper",
}


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

    scans = discover_stl_files(data_root)
    samples = sorted(
        identifier
        for identifier in {patient_id_for_scan(path, data_root) for path in scans}
        if identifier is not None
    )
    if not samples:
        raise RuntimeError(f"No lower/upper STL files found under {data_root}")

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


def discover_stl_files(root):
    paths = root.rglob("*")
    scans = sorted(
        path for path in paths if path.is_file() and path.suffix.lower() == ".stl"
    )
    if not scans:
        raise RuntimeError(f"No STL files found under {root}")
    return scans


def patient_id_for_scan(scan, data_root):
    relative = scan.relative_to(data_root)
    arch = infer_arch(relative)
    if arch is None:
        return None
    return infer_identifier(relative, arch)


def infer_arch(relative_path):
    text = relative_path.as_posix().casefold()
    for token, arch in ARCH_TOKENS.items():
        if token in text:
            return arch
    return None


def infer_identifier(relative_path, arch):
    parts = relative_path.with_suffix("").parts
    for part in reversed(parts[:-1]):
        if not contains_arch_token(part):
            return part

    stem_tokens = [
        token
        for token in re.split(r"[^A-Za-z0-9]+", relative_path.stem)
        if token and ARCH_TOKENS.get(token.casefold()) != arch
    ]
    numeric_tokens = [
        token for token in stem_tokens if any(char.isdigit() for char in token)
    ]
    if numeric_tokens:
        return numeric_tokens[-1]
    if stem_tokens:
        return stem_tokens[-1]
    raise RuntimeError(f"Could not infer identifier from {relative_path}")


def contains_arch_token(text):
    text = text.casefold()
    return any(token in text for token in ARCH_TOKENS)


if __name__ == "__main__":
    main()
