import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
import trimesh

from scannormalizer.scan_inference import load_normalizer, normalize_scan


def parse_args():
    parser = argparse.ArgumentParser(
        description="Orient every STL scan under an input directory and render QA plots."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--points", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sampling", choices=("fps", "random"), default="fps")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--plots-per-image", type=int, default=40)
    parser.add_argument("--render-mode", choices=("surface", "points"), default="points")
    parser.add_argument("--render-points", type=int, default=5000)
    parser.add_argument("--render-faces", type=int, default=10000)
    parser.add_argument("--point-size", type=float, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.plot_only:
        output_paths = find_stl_files(output_dir, exclude_dir=output_dir / "qa")
        if not output_paths:
            raise RuntimeError(f"No oriented STL scans found under {output_dir}")
        render_contact_sheets(
            output_paths,
            output_dir / "qa",
            output_dir,
            args.plots_per_image,
            args.render_points,
            args.render_faces,
            args.render_mode,
            args.point_size,
        )
        print(f"QA images saved under {output_dir / 'qa'}", flush=True)
        return

    if args.checkpoint is None:
        raise RuntimeError("--checkpoint is required unless --plot-only is used")

    scans = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".stl"
    )
    if not scans:
        raise RuntimeError(f"No STL scans found under {input_dir}")

    normalizer = load_normalizer(
        args.checkpoint,
        device=args.device,
        points=args.points,
        sampling=args.sampling,
    )

    output_paths = []
    for index, scan_path in enumerate(scans, start=1):
        relative_path = scan_path.relative_to(input_dir)
        output_path = output_dir / relative_path
        result = normalize_scan(
            scan_path,
            output_path,
            normalizer,
        )
        output_paths.append(output_path)
        print(
            f"[{index}/{len(scans)}] {relative_path} -> {output_path.relative_to(output_dir)} "
            f"| rotation_class {result.rotation_index}",
            flush=True,
        )

    render_contact_sheets(
        output_paths,
        output_dir / "qa",
        output_dir,
        args.plots_per_image,
        args.render_points,
        args.render_faces,
        args.render_mode,
        args.point_size,
    )
    print(f"oriented {len(output_paths)} scans into {output_dir}", flush=True)
    print(f"QA images saved under {output_dir / 'qa'}", flush=True)


def find_stl_files(root, exclude_dir=None):
    root = root.resolve()
    exclude_dir = exclude_dir.resolve() if exclude_dir is not None else None
    paths = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".stl":
            continue
        if exclude_dir is not None and exclude_dir in path.resolve().parents:
            continue
        paths.append(path)
    return sorted(paths)


def render_contact_sheets(
    mesh_paths,
    qa_dir,
    output_dir,
    plots_per_image,
    render_points,
    render_faces,
    render_mode,
    point_size,
):
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for QA plotting. Install requirements first."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qa_dir.mkdir(parents=True, exist_ok=True)
    plots_per_image = max(1, plots_per_image)
    page_count = math.ceil(len(mesh_paths) / plots_per_image)

    for page_index in range(page_count):
        page_paths = mesh_paths[
            page_index * plots_per_image : (page_index + 1) * plots_per_image
        ]
        columns = min(5, len(page_paths))
        rows = math.ceil(len(page_paths) / columns)
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.0 * columns, 3.6 * rows),
            subplot_kw={"projection": "3d"},
        )
        axes = np.asarray(axes, dtype=object).reshape(-1)

        for axis, mesh_path in zip(axes, page_paths):
            plot_mesh(
                axis,
                mesh_path,
                mesh_path.relative_to(output_dir),
                render_points,
                render_faces,
                render_mode,
                point_size,
            )

        for axis in axes[len(page_paths) :]:
            axis.set_axis_off()

        figure.tight_layout()
        figure.savefig(qa_dir / f"oriented_scans_{page_index + 1:03d}.png", dpi=160)
        plt.close(figure)


def plot_mesh(axis, mesh_path, title, render_points, render_faces, render_mode, point_size):
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    points = np.asarray(mesh.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        axis.set_title(f"{title}\n(no vertices)", fontsize=8)
        axis.set_axis_off()
        return

    if render_mode == "surface" and len(mesh.faces) > 0:
        plot_surface(axis, points, np.asarray(mesh.faces), render_faces)
    else:
        plot_points(axis, points, render_points, point_size)

    set_equal_axes(axis, points)
    draw_axes(axis, points)
    axis.set_title(str(title), fontsize=8)
    axis.view_init(elev=18, azim=-55)


def plot_surface(axis, vertices, faces, render_faces):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if len(faces) > render_faces:
        rng = np.random.default_rng(0)
        faces = faces[rng.choice(len(faces), size=render_faces, replace=False)]

    surface = Poly3DCollection(
        vertices[faces],
        facecolor="#d8d8d8",
        edgecolor="none",
        alpha=0.96,
    )
    surface.set_sort_zpos(0)
    axis.add_collection3d(surface)


def plot_points(axis, points, render_points, point_size):
    if len(points) > render_points:
        rng = np.random.default_rng(0)
        points = points[rng.choice(len(points), size=render_points, replace=False)]

    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=point_size,
        c=points[:, 2],
        cmap="viridis",
        alpha=0.85,
        linewidths=0,
    )


def set_equal_axes(axis, points):
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max((maximum - minimum).max() / 2.0, 1e-3)

    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("X", color="red")
    axis.set_ylabel("Y", color="green")
    axis.set_zlabel("Z", color="blue")


def draw_axes(axis, points):
    span = points.max(axis=0) - points.min(axis=0)
    length = max(float(span.max()) * 0.6, 0.5)

    axis.plot([-length, length], [0, 0], [0, 0], color="red", linewidth=1.4)
    axis.plot([0, 0], [-length, length], [0, 0], color="green", linewidth=1.4)
    axis.plot([0, 0], [0, 0], [-length, length], color="blue", linewidth=1.4)
    axis.text(length, 0, 0, "+X", color="red", fontsize=8)
    axis.text(0, length, 0, "+Y", color="green", fontsize=8)
    axis.text(0, 0, length, "+Z", color="blue", fontsize=8)


if __name__ == "__main__":
    main()
