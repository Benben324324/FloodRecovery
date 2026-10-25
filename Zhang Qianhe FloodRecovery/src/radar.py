"""Small, split-preserving SAR water-recognition experiment."""
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .download import ROOT, RAW, download_radar, sha256

SIZES = {'train': 96, 'val': 32, 'test': 64}
TOTALS = {'train': 1404, 'val': 468, 'test': 468}


def prepare_radar():
    manifest_path = RAW / 'radar_subset_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if all((ROOT / row['path']).exists() for row in manifest['files']):
            return manifest
    import libarchive
    archive_path = download_radar()
    rng = np.random.default_rng(42)
    selected = {s: set(rng.choice(TOTALS[s], n, replace=False).tolist()) for s, n in SIZES.items()}
    files = []
    with libarchive.file_reader(str(archive_path)) as archive:
        for entry in archive:
            match = re.search(r'/(train|val|test)/(image|label_1D)/(\d+)\.png$', entry.pathname)
            if not match:
                continue
            split, kind, ident = match.groups()
            if int(ident) not in selected[split]:
                continue
            path = RAW / 'radar' / split / kind / f'{int(ident):05}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b''.join(entry.get_blocks()))
            files.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    if len(files) != 2 * sum(SIZES.values()):
        raise ValueError('Missing radar image or label.')
    manifest = {'record': 'https://doi.org/10.5281/zenodo.7198950', 'seed': 42,
                'original_split_sizes': TOTALS, 'subset_sizes': SIZES,
                'selected_ids': {s: sorted(v) for s, v in selected.items()}, 'files': files}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_split(split):
    samples = []
    for path in sorted((RAW / 'radar' / split / 'image').glob('*.png')):
        label_path = RAW / 'radar' / split / 'label_1D' / path.name
        image = np.asarray(Image.open(path))
        label = np.asarray(Image.open(label_path))
        if image.shape != (256, 256) or label.shape != image.shape or image.ndim != 2:
            raise ValueError('Unexpected SAR patch dimensions.')
        if not np.isin(label, [0, 1]).all():
            raise ValueError('Unexpected binary label encoding.')
        samples.append((path.stem, image.astype(np.float32) / 255, label.astype(bool)))
    if len(samples) != SIZES[split]:
        raise ValueError('SAR subset size mismatch.')
    return samples


def features(image):
    mean3 = uniform_filter(image, size=3, mode='reflect')
    variance = np.maximum(uniform_filter(image**2, size=3, mode='reflect') - mean3**2, 0)
    return np.stack([image, mean3, np.sqrt(variance), uniform_filter(image, size=9, mode='reflect')], axis=-1)


def confusion(y, prediction):
    return np.array([(y & prediction).sum(), (~y & prediction).sum(),
                     (y & ~prediction).sum(), (~y & ~prediction).sum()], dtype=np.int64)


def metrics(counts):
    tp, fp, fn, tn = np.asarray(counts, dtype=float)
    return {'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1), 'iou': tp / max(tp + fp + fn, 1),
            'accuracy': (tp + tn) / max(tp + fp + fn + tn, 1)}


def evaluate(samples, predict):
    return np.array([confusion(label, predict(image)) for _, image, label in samples])


def run_benchmark():
    splits = {s: load_split(s) for s in SIZES}
    rng = np.random.default_rng(43)
    x, y = [], []
    for _, image, label in splits['train']:
        select = rng.choice(image.size, 2048, replace=False)
        x.append(features(image).reshape(-1, 4)[select]); y.append(label.ravel()[select])
    x, y = np.concatenate(x), np.concatenate(y)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=300, random_state=43))
    model.fit(x, y)
    thresholds = np.linspace(0.02, 0.98, 97)
    scores = [metrics(evaluate(splits['val'], lambda a: a < t).sum(axis=0))['iou'] for t in thresholds]
    intensity_threshold = float(thresholds[np.argmax(scores)])
    val_probs = [model.predict_proba(features(a).reshape(-1, 4))[:, 1].reshape(a.shape) for _, a, _ in splits['val']]
    prob_thresholds = np.linspace(0.1, 0.9, 17)
    scores_lr = [metrics(np.sum([confusion(label, p >= t) for (_, _, label), p in zip(splits['val'], val_probs)], axis=0))['iou']
                 for t in prob_thresholds]
    prob_threshold = float(prob_thresholds[np.argmax(scores_lr)])
    predictors = {'Intensity 0.50': lambda a: a < 0.5,
                  'Tuned intensity': lambda a: a < intensity_threshold,
                  'Local logistic': lambda a: model.predict_proba(features(a).reshape(-1, 4))[:, 1].reshape(a.shape) >= prob_threshold}
    rows, tile_rows = [], []
    for name, predictor in predictors.items():
        counts = evaluate(splits['test'], predictor)
        result = metrics(counts.sum(axis=0))
        bootstrap_rng = np.random.default_rng(44)
        boots = [metrics(counts[bootstrap_rng.integers(0, len(counts), len(counts))].sum(axis=0))['iou'] for _ in range(1000)]
        rows.append({'model': name, **result, 'iou_ci_low': np.quantile(boots, .025),
                     'iou_ci_high': np.quantile(boots, .975), 'test_tiles': len(counts)})
        for (ident, _, _), count in zip(splits['test'], counts):
            tile_rows.append({'model': name, 'tile': ident, **dict(zip(['tp', 'fp', 'fn', 'tn'], count)), **metrics(count)})
    table = pd.DataFrame(rows)
    out = ROOT / 'outputs' / 'tables'
    table.to_csv(out / 'radar_metrics.csv', index=False)
    pd.DataFrame(tile_rows).to_csv(out / 'radar_tile_metrics.csv', index=False)
    config = {'seed': 43, 'training_pixels': len(y), 'features': ['intensity', 'mean_3', 'std_3', 'mean_9'],
              'intensity_threshold': intensity_threshold, 'probability_threshold': prob_threshold,
              'validation_intensity_iou': max(scores), 'validation_logistic_iou': max(scores_lr),
              'test_pixels': sum(a.size for _, a, _ in splits['test']),
              'geographic_coordinates': 'not specified in the patch files',
              'split_sizes': SIZES, 'bootstrap_seed': 44, 'bootstrap_repetitions': 1000}
    (out / 'radar_config.json').write_text(json.dumps(config, indent=2))
    return table, config, splits, predictors
