# ScanNormalizer

Minimal scan orientation normalizer.

The repository keeps the user-facing single-scan entry point at the root:

```bash
python orient_scan.py /path/to/scan.stl --checkpoint runs/rotation/best.pt
```

Reusable Python code lives in `src/scannormalizer/`, auxiliary tools live in `scripts/`, and local data/results are expected under ignored `data/` and `runs/` folders.

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

The training task uses the consistently oriented scans as the canonical frame:

1. Load mesh vertices.
2. Center them and scale to the unit sphere.
3. Sample a fixed number of points with farthest point sampling.
4. Apply either no rotation or a 180-degree rotation around X, Y, or Z.
5. Predict which of the four rotation classes was applied.
6. Train with cross entropy on the rotation class.

At inference time, the scan is PCA-aligned first, saved as a `_pca` mesh, then the model predicts which 180-degree correction to apply.

Train:

```bash
python scripts/train.py --data-root data/input --fold-dir data/splits/fold_1 --output-dir runs/rotation
```

Create split files:

```bash
python scripts/create_splits.py --data-root data/input --output-dir data/splits
```

Generate evaluation ground-truth rotations:

```bash
python scripts/generate_eval_gt.py --input-dir data/input --gt-json data/gt/ground_truth.json --seed 42
```

During training, full evaluation runs before epoch 1 and after every epoch by default. It evaluates only the validation split from `fold_dir/val.txt`, loads those scan names from `data/input/`, reads GT matrices from `data/gt/ground_truth.json`, and writes all predicted matrices to one `json/predictions.json` file inside each run directory. Disable it with `--no-eval`, or override paths with `--eval-input-dir`, `--eval-split-file`, and `--eval-gt-json`.

Each training run creates a separate directory under `--output-dir` containing `last.pt`, `best.pt`, and the local Weights & Biases files. Training logs to the `ios_orientation` Weights & Biases project by default. Disable it with `--no-wandb`.

Orient one scan:

```bash
python orient_scan.py /path/to/scan.stl --checkpoint runs/rotation/best.pt --output-dir data/output
```

Orient a full input directory while preserving patient subfolders:

```bash
python scripts/batch_orient_scans.py --input-dir data/input --output-dir data/output --checkpoint runs/rotation/best.pt
```

The batch script writes oriented STL files under the same relative paths in `data/output/` and saves quick visual QA sheets under `data/output/qa/`. By default each QA image contains up to 10 scans with X/Y/Z axes drawn in red/green/blue.

Regenerate only the QA plots from already oriented scans:

```bash
python scripts/batch_orient_scans.py --output-dir data/output --plot-only
```

The QA plots render points by default. If needed, increase the point size or switch to slower surface rendering:

```bash
python scripts/batch_orient_scans.py --output-dir data/output --plot-only --point-size 2.0
python scripts/batch_orient_scans.py --output-dir data/output --plot-only --render-mode surface --render-faces 20000
```

Use the inference API from Python:

```python
from scannormalizer.scan_inference import load_normalizer, normalize_scan

normalizer = load_normalizer("runs/rotation/best.pt", device="cuda", points=4096)
result = normalize_scan("data/input/patient/lower.stl", "data/output/patient/lower.stl", normalizer)
print(result.rotation_index)
```
