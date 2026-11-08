from __future__ import annotations
import csv
from pathlib import Path

def read_metadata(data_dir: Path):
    metadata={}
    with (data_dir/'labels.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream, delimiter=';'): metadata[row['Label']]=row
    labels=sorted(metadata); return labels, [metadata[c]['English'] for c in labels], metadata

def iter_split(data_dir: Path, split: str):
    with (data_dir/f'x_{split}.txt').open(encoding='utf-8') as xs, (data_dir/f'y_{split}.txt').open(encoding='utf-8') as ys:
        for i,(text,label) in enumerate(zip(xs,ys,strict=True)): yield i,text.rstrip('\r\n'),label.strip()

