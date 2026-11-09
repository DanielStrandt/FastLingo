from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
from .accelerated import AcceleratedIdentifier

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['short','long'],required=True); p.add_argument('--iterations',type=int,default=1000); a=p.parse_args()
    model=AcceleratedIdentifier(Path(f'artifacts/lid235_{a.mode}.npz')); text='This is a short sentence for edge inference.' if a.mode=='short' else 'This is a representative language identification sentence used for edge inference.'
    model.predict(text); values=[]
    for _ in range(a.iterations):
        start=time.perf_counter(); model.predict(text); values.append((time.perf_counter()-start)*1000)
    values.sort(); out={'mode':a.mode,'iterations':a.iterations,'input_length':len(text),'p50_ms':statistics.median(values),'p95_ms':values[int(.95*len(values))-1],'p99_ms':values[int(.99*len(values))-1],'throughput_per_second':1000/statistics.fmean(values)}; print(json.dumps(out,indent=2))
if __name__=='__main__': main()

