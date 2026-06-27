import argparse
from pathlib import Path

import torch

from scan_inference import load_normalizer, normalize_scan

# checkpoint /homes/mlugli/ScanNormalizer/exp/rotation/20260618_093503/best.pt
def parse_args():
    parser = argparse.ArgumentParser(description="Orient one scan with a trained normalizer.")
    parser.add_argument("scan", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pca_output_path = output_dir / f"{args.scan.stem}_pca{args.scan.suffix}"
    output_path = output_dir / f"{args.scan.stem}_oriented{args.scan.suffix}"

    normalizer = load_normalizer(args.checkpoint, args.device, points=args.points)
    result = normalize_scan(
        args.scan,
        output_path,
        normalizer,
        pca_output_path=pca_output_path,
    )

    print(f"rotation logits: {result.logits}")
    print(f"selected rotation index: {result.rotation_index}")
    print(f"pca saved: {result.pca_output_path}")
    print(f"saved: {result.output_path}")


if __name__ == "__main__":
    main()
