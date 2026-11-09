from __future__ import annotations
import argparse, struct
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('input',type=Path); p.add_argument('output',type=Path); a=p.parse_args(); z=np.load(a.input,allow_pickle=False)
    labels=z['labels'].tolist(); names=z['label_names'].tolist(); proto=z['prototypes'].astype(np.int8); weights=z['channel_weights'].astype(np.float32); bias=z['class_bias'].astype(np.float32); dims=int(z['dimensions']); sparsity=int(z['sparsity']); nvals=tuple(z['n_values'].tolist()); seed=int(z['seed'])
    with a.output.open('wb') as f:
        f.write(b'FLNG1\0'); f.write(struct.pack('<IIIIQ',len(labels),proto.shape[0],dims,sparsity,seed)); f.write(struct.pack('<I',len(nvals))); f.write(bytes(nvals)); f.write(weights.tobytes()); f.write(bias.tobytes()); f.write(proto.tobytes())
        for label,name in zip(labels,names):
            for value in (label,name):
                data=value.encode('utf8'); f.write(struct.pack('<H',len(data))); f.write(data)
if __name__=='__main__': main()

