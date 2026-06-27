# ScanNormalizer

Minimal scan orientation normalizer.

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
python train.py --data-root /work/grana_maxillo/IOS_3DT --fold-dir splits/fold_1 --output-dir exp/rotation
```

Generate evaluation ground-truth rotations:

```bash
python generate_eval_gt.py --input-dir input --gt-json gt/ground_truth.json --seed 42
```

During training, full evaluation runs before epoch 1 and after every epoch by default. It evaluates only the validation split from `fold_dir/val.txt`, loads those scan names from `input/`, reads GT matrices from `gt/ground_truth.json`, and writes all predicted matrices to one `json/predictions.json` file inside each run directory. Disable it with `--no-eval`, or override paths with `--eval-input-dir`, `--eval-split-file`, and `--eval-gt-json`.

Each training run creates a separate directory under `--output-dir` containing `last.pt`, `best.pt`, and the local Weights & Biases files. Training logs to the `ios_orientation` Weights & Biases project by default. Disable it with `--no-wandb`.

Orient one scan:

```bash
python orient_scan.py /path/to/scan.stl --checkpoint exp/rotation/best.pt --output-dir output
```

Orient a full input directory while preserving patient subfolders:

```bash
python batch_orient_scans.py --input-dir input --output-dir output --checkpoint exp/rotation/best.pt
```

The batch script writes oriented STL files under the same relative paths in `output/` and saves quick visual QA sheets under `output/qa/`. By default each QA image contains up to 10 scans with X/Y/Z axes drawn in red/green/blue.

Regenerate only the QA plots from already oriented scans:

```bash
python batch_orient_scans.py --output-dir output --plot-only
```

The QA plots render points by default. If needed, increase the point size or switch to slower surface rendering:

```bash
python batch_orient_scans.py --output-dir output --plot-only --point-size 2.0
python batch_orient_scans.py --output-dir output --plot-only --render-mode surface --render-faces 20000
```

Use the inference API from Python:

```python
from scan_inference import load_normalizer, normalize_scan

normalizer = load_normalizer("exp/rotation/best.pt", device="cuda", points=4096)
result = normalize_scan("input/patient/lower.stl", "output/patient/lower.stl", normalizer)
print(result.rotation_index)
```
