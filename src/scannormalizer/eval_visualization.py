import html
import json
from pathlib import Path

import numpy as np
import trimesh


def build_eval_visualizations(
    predictions,
    input_dir,
    output_dir,
    epoch,
    worst_count=5,
    render_points=8000,
):
    if not predictions or worst_count <= 0:
        return {}

    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranked = sorted(predictions, key=lambda item: item["geodesic_loss"], reverse=True)
    worst = ranked[:worst_count]
    best = min(predictions, key=lambda item: item["geodesic_loss"])

    worst_path = output_dir / f"test_worst_epoch_{epoch:03d}.html"
    best_path = output_dir / f"test_low_error_epoch_{epoch:03d}.html"

    _write_scan_pairs_html(
        worst,
        input_dir,
        worst_path,
        f"Epoch {epoch:03d}: Highest Rotation Error Scans",
        render_points,
    )
    _write_scan_pairs_html(
        [best],
        input_dir,
        best_path,
        f"Epoch {epoch:03d}: Low Geodesic Error Scan",
        render_points,
    )

    return {
        "test/worst_rotation_error_scans_3d": worst_path,
        "test/low_geodesic_error_scan_3d": best_path,
    }


def _write_scan_pairs_html(items, input_dir, output_path, title, render_points):
    traces = []
    annotations = []
    layout = {
        "title": {"text": title},
        "height": max(520, 420 * len(items)),
        "margin": {"l": 0, "r": 0, "t": 80, "b": 0},
        "showlegend": True,
    }
    scan_pairs = []
    for item in items:
        original, before, after = _load_visualization_points(input_dir / item["scan"], item)
        original, before, after = _sample_points(original, before, after, render_points)
        scan_pairs.append((item, original, before, after))

    for row, (item, original, before, after) in enumerate(scan_pairs, start=1):
        original_scene = _add_scene(layout, annotations, row, 1, len(scan_pairs), _original_title(item))
        before_scene = _add_scene(layout, annotations, row, 2, len(scan_pairs), "After GT rotation")
        after_scene = _add_scene(layout, annotations, row, 3, len(scan_pairs), "After model normalization")

        traces.append(_scan_trace(original, original_scene, "Original"))
        traces.append(_scan_trace(before, before_scene, "Before"))
        traces.append(_scan_trace(after, after_scene, "After"))
        limit = max(_point_radius(original), _point_radius(before), _point_radius(after)) * 1.15
        traces.extend(_axis_traces(limit, original_scene, show_legend=row == 1))
        traces.extend(_axis_traces(limit, before_scene, show_legend=False))
        traces.extend(_axis_traces(limit, after_scene, show_legend=False))

    layout["annotations"] = annotations
    _write_plotly_html(output_path, traces, layout, title)


def _original_title(item):
    error_deg = np.degrees(item["geodesic_loss"])
    return f"Original: {item['scan']}<br>error {error_deg:.2f} deg"


def _add_scene(layout, annotations, row, col, rows, title):
    index = (row - 1) * 3 + col
    scene = "scene" if index == 1 else f"scene{index}"
    x_domains = [[0.0, 0.32], [0.34, 0.66], [0.68, 1.0]]
    x_domain = x_domains[col - 1]
    row_height = 1.0 / rows
    y_domain = [1.0 - row * row_height + 0.03, 1.0 - (row - 1) * row_height - 0.05]
    y_domain = [max(0.0, y_domain[0]), min(1.0, y_domain[1])]

    layout[scene] = {
        "domain": {"x": x_domain, "y": y_domain},
        "aspectmode": "cube",
        "xaxis": {"title": {"text": "X"}, "color": "red"},
        "yaxis": {"title": {"text": "Y"}, "color": "green"},
        "zaxis": {"title": {"text": "Z"}, "color": "blue"},
        "camera": {"eye": {"x": 1.5, "y": -1.8, "z": 1.2}},
    }
    annotations.append(
        {
            "text": title,
            "x": sum(x_domain) / 2.0,
            "y": min(y_domain[1] + 0.035, 1.0),
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": {"size": 13},
        }
    )
    return scene


def _load_visualization_points(scan_path, item):
    mesh = trimesh.load(scan_path, force="mesh", process=False)
    points = np.asarray(mesh.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise RuntimeError(f"Could not load vertices from {scan_path}")

    input_rotation = np.asarray(item["input_rotation_matrix"], dtype=np.float32)
    predicted_correction = np.asarray(item["matrix"], dtype=np.float32)

    original = _normalize_points(points)
    before = _normalize_points(points @ input_rotation)
    after = before @ predicted_correction
    return original, before, after


def _normalize_points(points):
    centered = points - points.mean(axis=0, keepdims=True)
    scale = max(float(np.linalg.norm(centered, axis=1).max()), 1e-8)
    return centered / scale


def _sample_points(original, before, after, count):
    if len(before) <= count:
        return original, before, after

    rng = np.random.default_rng(0)
    indices = rng.choice(len(before), size=count, replace=False)
    return original[indices], before[indices], after[indices]


def _scan_trace(points, scene, name):
    return {
        "type": "scatter3d",
        "scene": scene,
        "x": points[:, 0].tolist(),
        "y": points[:, 1].tolist(),
        "z": points[:, 2].tolist(),
        "mode": "markers",
        "name": name,
        "showlegend": False,
        "marker": {
            "size": 1.6,
            "color": points[:, 2].tolist(),
            "colorscale": "Viridis",
            "opacity": 0.86,
        },
    }


def _axis_traces(limit, scene, show_legend):
    axes = [
        ("X", "red", [-limit, limit], [0, 0], [0, 0], ["", "+X"]),
        ("Y", "green", [0, 0], [-limit, limit], [0, 0], ["", "+Y"]),
        ("Z", "blue", [0, 0], [0, 0], [-limit, limit], ["", "+Z"]),
    ]
    traces = []
    for name, color, x, y, z, text in axes:
        traces.append(
            {
                "type": "scatter3d",
                "scene": scene,
                "x": x,
                "y": y,
                "z": z,
                "mode": "lines+text",
                "name": f"{name} axis",
                "legendgroup": f"{name} axis",
                "showlegend": show_legend,
                "line": {"color": color, "width": 7},
                "text": text,
                "textfont": {"color": color, "size": 14},
                "hoverinfo": "skip",
            }
        )
    return traces


def _write_plotly_html(output_path, traces, layout, title):
    figure_json = json.dumps({"data": traces, "layout": layout}).replace("</", "<\\/")
    output_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body style="margin:0;">
  <div id="scan-plot" style="width:100%;height:100vh;"></div>
  <script>
    const figure = {figure_json};
    Plotly.newPlot('scan-plot', figure.data, figure.layout, {{responsive: true}});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def _point_radius(points):
    return max(float(np.abs(points).max()), 1e-3)
