import json
from pathlib import Path

import torch

from .geodesic_loss import GeodesicLoss
from .scan_inference import Normalizer, predict_normalization_matrix


def run_evaluation(
    model,
    device,
    input_dir,
    gt_json,
    predictions_json,
    epoch,
    points,
    sampling,
    scan_files=None,
):
    input_dir = Path(input_dir).expanduser().resolve()
    gt_json = Path(gt_json).expanduser().resolve()
    predictions_json = Path(predictions_json)
    if scan_files is None:
        scans = sorted(path for path in input_dir.rglob("*.stl") if path.is_file())
    else:
        scans = sorted(Path(path).expanduser().resolve() for path in scan_files)
    if not scans:
        raise RuntimeError(f"No test STL files found under {input_dir}")
    missing = [path for path in scans if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing test scan: {missing[0]}")

    was_training = model.training
    model.eval()
    normalizer = Normalizer(model=model, device=device, points=points, sampling=sampling)
    gt_by_scan = load_gt_matrices(gt_json)

    predictions = []
    with torch.no_grad():
        for index, scan in enumerate(scans, start=1):
            relative_scan = scan.relative_to(input_dir).as_posix()
            if relative_scan not in gt_by_scan:
                raise RuntimeError(f"Missing GT rotation for test scan: {relative_scan}")
            gt_rotation = gt_by_scan[relative_scan]
            print(
                f"  test scan {index}/{len(scans)}: {relative_scan}",
                flush=True,
            )
            result = predict_normalization_matrix(
                scan,
                normalizer,
                input_rotation=gt_rotation,
            )
            predictions.append(
                {
                    "scan": relative_scan,
                    "input_rotation_matrix": gt_rotation,
                    "matrix": result.matrix,
                    "rotation_index": result.rotation_index,
                    "logits": result.logits,
                }
            )

    write_predictions_json(predictions_json, epoch, predictions)
    mean_loss, losses = compute_geodesic_from_json(predictions_json, gt_json, epoch)
    for item, loss in zip(predictions, losses):
        item["geodesic_loss"] = loss
    write_predictions_json(predictions_json, epoch, predictions, mean_loss)

    if was_training:
        model.train()

    return {
        "json_path": predictions_json,
        "mean_geodesic_loss": mean_loss,
        "geodesic_losses": losses,
        "predictions": predictions,
    }


def compute_geodesic_from_json(predictions_json_path, gt_json_path, epoch=None):
    data = json.loads(Path(predictions_json_path).read_text())
    gt_by_scan = load_gt_matrices(gt_json_path)
    epoch_data = select_epoch(data["epochs"], epoch)
    predicted = []
    targets = []

    for item in epoch_data["predictions"]:
        gt = gt_by_scan[item["scan"]]
        predicted.append(item["matrix"])
        targets.append(torch.tensor(gt, dtype=torch.float32).T)

    predicted = torch.tensor(predicted, dtype=torch.float32)
    targets = torch.stack(targets)
    losses = GeodesicLoss(reduction="none")(targets, predicted).tolist()
    return sum(losses) / max(len(losses), 1), losses


def load_gt_matrices(gt_json_path):
    data = json.loads(Path(gt_json_path).read_text())
    return {item["scan"]: item["rotation_matrix"] for item in data["rotations"]}


def select_epoch(epochs, epoch):
    if epoch is None:
        return epochs[-1]
    for item in epochs:
        if item["epoch"] == epoch:
            return item
    raise RuntimeError(f"No predictions found for epoch {epoch}")


def write_predictions_json(path, epoch, predictions, mean_loss=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text()) if path.exists() else {"epochs": []}

    epoch_data = {
        "epoch": epoch,
        "predictions": predictions,
    }
    if mean_loss is not None:
        epoch_data["mean_geodesic_loss"] = mean_loss

    data["epochs"] = [item for item in data["epochs"] if item["epoch"] != epoch]
    data["epochs"].append(epoch_data)
    data["epochs"].sort(key=lambda item: item["epoch"])

    path.write_text(json.dumps(data, indent=2) + "\n")
