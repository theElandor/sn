from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh

from .dataset import farthest_point_sample, normalize_points, pca_orient_points, random_point_sample
from .model import RotationNormalizer


@dataclass
class Normalizer:
    model: RotationNormalizer
    device: str
    points: int
    sampling: str


@dataclass
class NormalizationResult:
    output_path: Path
    rotation_index: int
    logits: list[float]
    matrix: list[list[float]]
    pca_output_path: Path | None = None


@dataclass
class MatrixPrediction:
    matrix: list[list[float]]
    rotation_index: int
    logits: list[float]


def load_normalizer(checkpoint_path, device=None, points=None, sampling="fps"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sample_count = points or checkpoint.get("points")

    model = RotationNormalizer().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return Normalizer(
        model=model,
        device=device,
        points=sample_count,
        sampling=sampling,
    )


def normalize_scan(scan_path, output_path, normalizer, pca_output_path=None):
    scan_path = Path(scan_path)
    output_path = Path(output_path)
    pca_output_path = Path(pca_output_path) if pca_output_path is not None else None

    inference = _run_inference(scan_path, normalizer)
    mesh = inference["mesh"]
    pca_vertices = inference["pca_vertices"]
    rotation = inference["rotation"]

    if pca_output_path is not None:
        pca_output_path.parent.mkdir(parents=True, exist_ok=True)
        pca_mesh = mesh.copy()
        pca_mesh.vertices = pca_vertices.numpy()
        pca_mesh.export(pca_output_path)

    oriented_vertices = pca_vertices @ rotation.T
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.vertices = oriented_vertices.numpy()
    mesh.export(output_path)

    return NormalizationResult(
        output_path=output_path,
        rotation_index=inference["rotation_index"],
        logits=inference["logits"].tolist(),
        matrix=inference["matrix"].tolist(),
        pca_output_path=pca_output_path,
    )


def predict_normalization_matrix(scan_path, normalizer, input_rotation=None):
    inference = _run_inference(scan_path, normalizer, input_rotation=input_rotation)
    return MatrixPrediction(
        matrix=inference["matrix"].tolist(),
        rotation_index=inference["rotation_index"],
        logits=inference["logits"].tolist(),
    )


def _run_inference(scan_path, normalizer, input_rotation=None):
    mesh = trimesh.load(scan_path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise RuntimeError(f"Could not load vertices from {scan_path}")

    points = torch.from_numpy(vertices)
    if input_rotation is not None:
        rotation = torch.as_tensor(input_rotation, dtype=points.dtype)
        points = points @ rotation

    normalized, center, scale = normalize_points(points)
    if normalizer.sampling == "fps":
        sampled = farthest_point_sample(normalized, normalizer.points)
    else:
        sampled = random_point_sample(normalized, normalizer.points)

    pca_sampled, basis = pca_orient_points(sampled)
    pca_sampled = pca_sampled.unsqueeze(0).to(normalizer.device)

    with torch.no_grad():
        output = normalizer.model(pca_sampled)
        logits = output["logits"][0].cpu()
        rotation_index = logits.argmax().item()
        rotation = output["predicted_rotation"][0].cpu()

    normalized_vertices = (points - center) / scale
    pca_vertices = normalized_vertices @ basis
    matrix = basis @ rotation.T

    return {
        "mesh": mesh,
        "pca_vertices": pca_vertices,
        "rotation": rotation,
        "matrix": matrix,
        "rotation_index": rotation_index,
        "logits": logits,
    }
