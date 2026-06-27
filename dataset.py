import math
import random
from pathlib import Path

import fpsample
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from model import ROTATION_CLASS_MATRICES


MESH_EXTENSIONS = {".stl", ".ply", ".obj", ".off"}


class ScanDataset(Dataset):
    def __init__(
        self,
        root,
        points=16000,
        sampling="random",
        max_fps_candidates=4096,
        files=None,
        pca_validation=False,
    ):
        self.root = Path(root).expanduser()
        self.points = points
        self.sampling = sampling
        self.max_fps_candidates = max_fps_candidates
        self.pca_validation = pca_validation
        self.files = (
            [Path(path) for path in files]
            if files is not None
            else sorted(
                path for path in self.root.rglob("*") if path.suffix.lower() in MESH_EXTENSIONS
            )
        )
        if not self.files:
            raise RuntimeError(f"No mesh files found under {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        points = load_points(self.files[index])
        points, _, _ = normalize_points(points)
        if self.sampling == "fps":
            target = farthest_point_sample(points, self.points, self.max_fps_candidates)
        else:
            target = random_point_sample(points, self.points)

        if self.pca_validation:
            random_rotation = random_rotation_matrix(dtype=target.dtype, device=target.device)
            rotated = target @ random_rotation
            distorted, _ = pca_orient_points(rotated)
            rotation_class = best_rotation_class(distorted, target)
        else:
            rotation_class = random.randrange(ROTATION_CLASS_MATRICES.shape[0])
            distorted = target @ ROTATION_CLASS_MATRICES[rotation_class]

        return distorted, torch.tensor(rotation_class, dtype=torch.long), target


def load_points(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    points = np.asarray(mesh.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise RuntimeError(f"Could not load vertices from {path}")
    return torch.from_numpy(points)


def normalize_points(points):
    center = points.mean(dim=0, keepdim=True)
    centered = points - center
    scale = centered.norm(dim=1).max().clamp_min(1e-8)
    return centered / scale, center.squeeze(0), scale


def pca_orient_points(points):
    centered = points - points.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / max(points.shape[0] - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    basis = eigenvectors[:, order]

    if torch.det(basis) < 0:
        basis[:, -1] = -basis[:, -1]

    return points @ basis, basis


def random_rotation_matrix(dtype=torch.float32, device=None):
    axis = torch.randn(3, dtype=dtype, device=device)
    axis = axis / axis.norm().clamp_min(1e-8)
    x, y, z = axis.unbind()
    angle = torch.rand((), dtype=dtype, device=device) * (2.0 * math.pi)
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    one_minus_cos = 1.0 - cos_angle

    return torch.stack(
        [
            torch.stack(
                [
                    cos_angle + x * x * one_minus_cos,
                    x * y * one_minus_cos - z * sin_angle,
                    x * z * one_minus_cos + y * sin_angle,
                ]
            ),
            torch.stack(
                [
                    y * x * one_minus_cos + z * sin_angle,
                    cos_angle + y * y * one_minus_cos,
                    y * z * one_minus_cos - x * sin_angle,
                ]
            ),
            torch.stack(
                [
                    z * x * one_minus_cos - y * sin_angle,
                    z * y * one_minus_cos + x * sin_angle,
                    cos_angle + z * z * one_minus_cos,
                ]
            ),
        ]
    )


def best_rotation_class(points, target):
    matrices = ROTATION_CLASS_MATRICES.to(dtype=points.dtype, device=points.device)
    candidates = points.unsqueeze(0) @ matrices.transpose(1, 2)
    errors = torch.mean((candidates - target.unsqueeze(0)) ** 2, dim=(1, 2))
    return torch.argmin(errors).item()


def random_point_sample(points, count):
    if points.shape[0] >= count:
        choice = torch.randperm(points.shape[0])[:count]
    else:
        choice = torch.randint(points.shape[0], (count,))
    return points[choice]


def farthest_point_sample(points, count, max_candidates=64000):
    if points.shape[0] > max_candidates:
        choice = torch.randperm(points.shape[0])[:max_candidates]
        points = points[choice]

    sample_count = min(count, points.shape[0])
    points_np = np.ascontiguousarray(points.detach().cpu().numpy(), dtype=np.float32)
    selected = fpsample.bucket_fps_kdline_sampling(points_np, sample_count, h=3)
    selected = torch.as_tensor(selected, dtype=torch.long, device=points.device)
    sampled = points[selected]

    if sampled.shape[0] < count:
        extra = torch.randint(sampled.shape[0], (count - sampled.shape[0],), device=points.device)
        sampled = torch.cat([sampled, sampled[extra]], dim=0)

    return sampled
