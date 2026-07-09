import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import debugpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from scannormalizer.dataset import ScanDataset
from scannormalizer.eval_visualization import build_eval_visualizations
from scannormalizer.evaluate import run_evaluation
from scannormalizer.model import RotationNormalizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a scan rotation normalizer.")
    parser.add_argument("--data-root", required=True, help="Directory containing scan meshes.")
    parser.add_argument("--fold-dir", required=True, help="Directory containing train.txt and val.txt.")
    parser.add_argument("--output-dir", default="runs/rotation", help="Root for per-run outputs.")
    parser.add_argument("--points", type=int, default=16000, help="Points sampled per scan.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint to load model weights from before training.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sampling", choices=("random", "fps"), default="fps")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test-input-dir", default=None)
    parser.add_argument("--test-gt-json", default="data/gt/ground_truth.json")
    parser.add_argument("--test-sampling", choices=("random", "fps"), default="fps")
    parser.add_argument("--test-points", type=int, default=None)
    parser.add_argument("--test-visualize-count", type=int, default=5)
    parser.add_argument("--test-visualize-points", type=int, default=8000)
    parser.add_argument("--no-test", action="store_true")
    parser.add_argument("--wandb-project", default="ios_orientation")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with debugpy.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.debug == True:
        print("Hello")
        debugpy.listen(("0.0.0.0", 5681))
        print(">>> Debugger is listening on port 5681. Waiting for client to attach...")
        debugpy.wait_for_client()
        print(">>> Debugger attached. Resuming execution.")
    model = RotationNormalizer().to(args.device)
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"model parameters {model_parameters:,}", flush=True)

    run_dir = create_run_dir(Path(args.output_dir), args.wandb_name)
    checkpoint_path = load_pretrained_weights(model, args.checkpoint, args.device)
    save_run_config(run_dir, args, checkpoint_path)
    data_root = Path(args.data_root).expanduser()
    fold_dir = Path(args.fold_dir).expanduser()

    data_candidates = discover_stl_files(data_root)
    train_files = read_split_files(fold_dir / "train.txt", data_candidates)
    val_files = read_split_files(fold_dir / "val.txt", data_candidates)
    if not args.no_test:
        test_split_file = select_test_split_file(fold_dir)
        test_root = Path(args.test_input_dir).expanduser() if args.test_input_dir else data_root
        args.test_input_dir = str(test_root)
        test_candidates = (
            data_candidates
            if test_root.resolve() == data_root.resolve()
            else discover_stl_files(test_root)
        )
        args.test_files = [
            str(path) for path in read_split_files(test_split_file, test_candidates)
        ]
    train_set = ScanDataset(
        data_root,
        points=args.points,
        sampling=args.sampling,
        files=train_files,
    )
    val_set = ScanDataset(
        data_root,
        points=args.points,
        sampling=args.sampling,
        files=val_files,
        pca_validation=True,
    )
    print(
        f"fold {fold_dir} | train {len(train_set)} | val {len(val_set)} | "
        f"test {len(args.test_files) if not args.no_test else 0} | "
        f"sampling {args.sampling} | device {args.device} | run_dir {run_dir}",
        flush=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or run_dir.name,
            dir=str(run_dir),
            config={**vars(args), "run_dir": str(run_dir)},
        )
        wandb.define_metric("train/batch_step")
        wandb.define_metric("train/batch_loss", step_metric="train/batch_step")
        wandb.define_metric("train/running_epoch_loss", step_metric="train/batch_step")
        wandb.define_metric("epoch")
        wandb.define_metric("train/epoch_loss", step_metric="epoch")
        wandb.define_metric("train/epoch_accuracy", step_metric="epoch")
        wandb.define_metric("val/loss", step_metric="epoch")
        wandb.define_metric("val/mse_loss", step_metric="epoch")
        wandb.define_metric("val/accuracy", step_metric="epoch")
        wandb.define_metric("val/best_loss", step_metric="epoch")
        wandb.define_metric("test/geodesic_loss", step_metric="epoch")
        wandb.watch(model, log="gradients", log_freq=100)

    if not args.no_test:
        test_result = test_current_model(
            model,
            args,
            run_dir,
            epoch=0,
            visualize=not args.no_wandb,
        )
        log_test(test_result, epoch=0, use_wandb=not args.no_wandb)

    train_batch_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch:03d} training...", flush=True)
        train_loss, train_accuracy, train_batch_step = run_epoch(
            model,
            train_loader,
            args.device,
            optimizer,
            "train",
            args.log_interval,
            epoch=epoch,
            use_wandb=not args.no_wandb,
            train_batch_step=train_batch_step,
        )
        print(f"epoch {epoch:03d} validating...", flush=True)
        val_loss, val_accuracy, _ = run_epoch(
            model, val_loader, args.device, None, "val", args.log_interval
        )

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        print(
            f"epoch {epoch:03d} | train_ce_loss {train_loss:.6f} | "
            f"train_acc {train_accuracy:.4f} | val_mse_loss {val_loss:.6f} | "
            f"val_acc {val_accuracy:.4f}",
            flush=True,
        )

        if not args.no_test:
            test_result = test_current_model(
                model,
                args,
                run_dir,
                epoch,
                visualize=not args.no_wandb,
            )
            log_test(test_result, epoch, use_wandb=not args.no_wandb)

        if not args.no_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "train/epoch_accuracy": train_accuracy,
                    "val/loss": val_loss,
                    "val/accuracy": val_accuracy,
                    "val/best_loss": best_val,
                    "train/cross_entropy_loss": train_loss,
                    "val/mse_loss": val_loss,
                    "val/best_mse_loss": best_val,
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )

        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "points": args.points,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if is_best:
            torch.save(checkpoint, run_dir / "best.pt")

    if not args.no_wandb:
        wandb.finish()


def create_run_dir(output_root, run_name=None):
    output_root.mkdir(parents=True, exist_ok=True)
    base_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(
        char if char.isalnum() or char in "._-" else "_" for char in base_name
    ).strip("._-")
    if not safe_name:
        safe_name = "run"

    for suffix in range(1000):
        name = safe_name if suffix == 0 else f"{safe_name}_{suffix:03d}"
        run_dir = output_root / name
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            continue

    raise RuntimeError(f"Could not create a unique run directory under {output_root}")


def load_pretrained_weights(model, checkpoint_path, device):
    if checkpoint_path is None:
        return None

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_model_weights(checkpoint)
    model.load_state_dict(state_dict)
    print(f"loaded pretrained model weights only from {checkpoint_path}", flush=True)
    return checkpoint_path


def extract_model_weights(checkpoint):
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def save_run_config(run_dir, args, checkpoint_path):
    config = {
        "command": sys.argv,
        "args": vars(args),
        "pretrained_checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }
    (run_dir / "args.json").write_text(json.dumps(config, indent=2) + "\n")


def discover_stl_files(root):
    root = Path(root).expanduser()
    paths = root.rglob("*")
    files = sorted(
        path.resolve()
        for path in paths
        if path.is_file() and path.suffix.lower() == ".stl"
    )
    if not files:
        raise RuntimeError(f"No STL files found under {root}")
    return files


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


def select_test_split_file(fold_dir):
    test_split = fold_dir / "test.txt"
    if test_split.exists():
        return test_split
    return fold_dir / "val.txt"


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
        formatted = format_matches(matches)
        raise RuntimeError(
            f"Split tuple '{identifier} {arch}' from {split_path}:{line_number} "
            f"matched multiple STL files: {formatted}"
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


def test_current_model(model, args, run_dir, epoch, visualize=False):
    print(f"epoch {epoch:03d} testing held-out split...", flush=True)
    result = run_evaluation(
        model=model,
        device=args.device,
        input_dir=args.test_input_dir,
        gt_json=args.test_gt_json,
        predictions_json=run_dir / "json" / "predictions.json",
        epoch=epoch,
        points=args.test_points or args.points,
        sampling=args.test_sampling,
        scan_files=args.test_files,
    )
    print(
        f"epoch {epoch:03d} | test_geodesic_loss "
        f"{result['mean_geodesic_loss']:.6f} | json {result['json_path']}",
        flush=True,
    )

    if visualize:
        result["visualizations"] = build_eval_visualizations(
            result["predictions"],
            args.test_input_dir,
            run_dir / "html" / f"epoch_{epoch:03d}",
            epoch,
            worst_count=args.test_visualize_count,
            render_points=args.test_visualize_points,
        )
    return result


def log_test(result, epoch, use_wandb):
    if not use_wandb:
        return

    rows = [
        [index, item["scan"], item["geodesic_loss"]]
        for index, item in enumerate(result["predictions"])
    ]
    table = wandb.Table(data=rows, columns=["scan_index", "scan", "geodesic_loss"])
    payload = {
        "epoch": epoch,
        "test/geodesic_loss": result["mean_geodesic_loss"],
        "test/geodesic_values": wandb.plot.line(
            table,
            "scan_index",
            "geodesic_loss",
            title=f"Test Geodesic Losses Epoch {epoch:03d}",
        ),
    }
    for name, path in result.get("visualizations", {}).items():
        payload[name] = wandb.Html(str(path))

    wandb.log(payload)


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    phase="train",
    log_interval=10,
    epoch=None,
    use_wandb=False,
    train_batch_step=0,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (distorted, rotation_class, target) in enumerate(loader, start=1):
        distorted = distorted.to(device, non_blocking=True)
        rotation_class = rotation_class.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            output = model(distorted)
            if training:
                loss = F.cross_entropy(output["logits"], rotation_class)
            else:
                corrected = distorted @ output["predicted_rotation"].transpose(1, 2)
                loss = F.mse_loss(corrected, target)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * distorted.shape[0]
        total_correct += (output["logits"].argmax(dim=1) == rotation_class).sum().item()
        total_samples += distorted.shape[0]
        running_loss = total_loss / total_samples
        running_accuracy = total_correct / total_samples

        if use_wandb and training:
            train_batch_step += 1
            wandb.log(
                {
                    "epoch": epoch,
                    "train/batch_step": train_batch_step,
                    "train/batch_loss": loss.item(),
                    "train/running_epoch_loss": running_loss,
                    "train/running_epoch_accuracy": running_accuracy,
                }
            )

        if log_interval and batch_idx % log_interval == 0:
            loss_name = "ce_loss" if training else "mse_loss"
            print(
                f"  {phase} batch {batch_idx}/{len(loader)} | {loss_name} {running_loss:.6f} | "
                f"acc {running_accuracy:.4f}",
                flush=True,
            )

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1), train_batch_step


if __name__ == "__main__":
    main()
