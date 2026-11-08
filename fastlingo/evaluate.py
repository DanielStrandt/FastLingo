from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from .data import iter_split, read_metadata
from .model import CompactIdentifier

def manifest_rows(path: Path):
    with path.open(encoding='utf8') as f:
        for line in f:
            yield json.loads(line)

def short_test_rows(path: Path, labels: set[str]):
    counts = Counter()
    for row in manifest_rows(path):
        if row['source'] not in ('tatoeba', 'wili') or row['label'] not in labels or not row['text'].strip():
            continue
        cap = 400 if row['source'] == 'tatoeba' else 300
        if counts[row['label']] >= cap:
            continue
        counts[row['label']] += 1
        key = hashlib.blake2b((row['label'] + '\0' + row['text']).encode('utf8'), digest_size=2).digest()
        if int.from_bytes(key, 'big') % 10 == 0:
            yield row['text'], row['label']

def metrics(y_true, y_pred, labels):
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    f1 = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {'examples': len(y_true), 'accuracy': correct / max(1, len(y_true)), 'macro_f1': sum(f1) / max(1, len(f1))}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['short', 'long'], required=True)
    p.add_argument('--model', type=Path); p.add_argument('--wili-dir', type=Path, default=Path('data/raw/wili-2018'))
    p.add_argument('--manifest', type=Path, default=Path('data/mixture/manifest.jsonl')); p.add_argument('--report', type=Path)
    a = p.parse_args(); a.model = a.model or Path(f'artifacts/lid235_{a.mode}.npz'); a.report = a.report or Path(f'reports/{a.mode}_test.json')
    labels, _, _ = read_metadata(a.wili_dir); label_set = set(labels); model = CompactIdentifier(a.model)
    if a.mode == 'long':
        rows = ((text, label) for _, text, label in iter_split(a.wili_dir, 'test') if label in label_set)
    else:
        rows = short_test_rows(a.manifest, label_set)
    y_true, y_pred = [], []
    for text, label in rows:
        y_true.append(label); y_pred.append(model.predict(text, top_k=1)['label'])
    report = {'mode': a.mode, 'model': str(a.model), 'labels': len(labels), **metrics(y_true, y_pred, labels)}
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(report, indent=2), encoding='utf8'); print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

