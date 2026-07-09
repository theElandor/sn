import argparse
import sys
from pathlib import Path
import debugpy

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from scannormalizer.scan_inference import load_normalizer, normalize_scan, transform_scan


def parse_args():
    parser = argparse.ArgumentParser(description="Orient one scan with a trained normalizer.")
    parser.add_argument("scan", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="data/output")
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with debugpy.")
    parser.add_argument(
        "--orient-only",
        action="store_true",
        help="Apply only the predicted orientation matrix; do not center or scale the output to the unit sphere.",
    )
    parser.add_argument(
        "--preserve-occlusion",
        action="store_true",
        help="Treat the scan argument as a patient folder, or a scan inside one, and transform sibling lower.stl and upper.stl together.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug == True:
        print("Hello, happy debugging.")
        debugpy.listen(("0.0.0.0", 5681))
        print(">>> Debugger is listening on port 5681. Waiting for client to attach...")
        debugpy.wait_for_client()
        print(">>> Debugger attached. Resuming execution.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer = load_normalizer(args.checkpoint, args.device, points=args.points)

    if args.preserve_occlusion:
        patient_dir = args.scan if args.scan.is_dir() else args.scan.parent
        lower_path = find_patient_scan(patient_dir, "lower")
        upper_path = find_patient_scan(patient_dir, "upper")
        lower_output_path = output_dir / f"{lower_path.stem}_oriented{lower_path.suffix}"
        upper_output_path = output_dir / f"{upper_path.stem}_oriented{upper_path.suffix}"
        pca_output_path = output_dir / f"{lower_path.stem}_pca{lower_path.suffix}"
        result = normalize_scan(
            lower_path,
            lower_output_path,
            normalizer,
            pca_output_path=pca_output_path,
            orient_only=args.orient_only,
        )
        transform_scan(
            upper_path,
            upper_output_path,
            result.matrix,
            center=result.center,
            scale=result.scale,
            orient_only=args.orient_only,
        )

        print(f"rotation logits: {result.logits}")
        print(f"selected rotation index: {result.rotation_index}")
        print(f"pca saved: {result.pca_output_path}")
        print(f"lower saved: {result.output_path}")
        print(f"upper saved: {upper_output_path}")
        return

    pca_output_path = output_dir / f"{args.scan.stem}_pca{args.scan.suffix}"
    output_path = output_dir / f"{args.scan.stem}_oriented{args.scan.suffix}"
    result = normalize_scan(
        args.scan,
        output_path,
        normalizer,
        pca_output_path=pca_output_path,
        orient_only=args.orient_only,
    )

    print(f"rotation logits: {result.logits}")
    print(f"selected rotation index: {result.rotation_index}")
    print(f"pca saved: {result.pca_output_path}")
    print(f"saved: {result.output_path}")


def find_patient_scan(patient_dir, scan_type):
    matches = []
    for path in Path(patient_dir).iterdir():
        if not path.is_file() or path.suffix.lower() != ".stl":
            continue
        classified_type = classify_scan(path)
        if classified_type == scan_type:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {scan_type} scan under {patient_dir}")
    return matches[0]


def classify_scan(scan_path):
    name = scan_path.name.lower()
    is_lower = "lower" in name or "mandibular" in name
    is_upper = "upper" in name or "maxillary" in name
    if is_lower and is_upper:
        raise RuntimeError(f"Ambiguous lower/upper scan name: {scan_path}")
    if is_lower:
        return "lower"
    if is_upper:
        return "upper"
    return None


if __name__ == "__main__":
    main()
