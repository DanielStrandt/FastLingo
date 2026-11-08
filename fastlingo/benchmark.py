"""Detailed standalone benchmarks for the compact HDC artifacts."""
from __future__ import annotations
import argparse, json, os, platform, statistics, time, tracemalloc
from pathlib import Path
from .model import CompactIdentifier, encode

TEXT = "This is a representative language identification sentence used for edge inference."

def summarize(values):
    ordered = sorted(values)
    return {'p50_ms': statistics.median(values), 'p95_ms': ordered[max(0, int(.95 * len(ordered)) - 1)], 'p99_ms': ordered[max(0, int(.99 * len(ordered)) - 1)], 'mean_ms': statistics.fmean(values), 'min_ms': min(values), 'max_ms': max(values), 'throughput_per_second': 1000.0 / statistics.fmean(values)}

def timed(fn, iterations, warmup):
    for _ in range(warmup): fn()
    values = []
    for _ in range(iterations):
        start = time.perf_counter(); fn(); values.append((time.perf_counter() - start) * 1000)
    return summarize(values)

def profile(path: Path, mode: str, iterations: int):
    load_start = time.perf_counter(); model = CompactIdentifier(path); load_ms = (time.perf_counter() - load_start) * 1000
    lengths = [16, 32, 64, 128, 256, 512, 1024, 2048] if mode == 'long' else [8, 16, 32, 64, 128, 240]
    latency = {}
    for length in lengths:
        text = ((TEXT + ' ') * ((length // (len(TEXT) + 1)) + 2))[:length]
        latency[str(length)] = {'characters': len(text), 'predict': timed(lambda: model.predict(text), iterations, min(20, max(3, iterations // 10))), 'encode_only': timed(lambda: encode(text, dimensions=model.dimensions, sparsity=model.sparsity, n_values=model.n_values, seed=model.seed), iterations, min(20, max(3, iterations // 10)))}
    tracemalloc.start(); model.predict(TEXT); _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return {'mode': mode, 'model': str(path), 'model_bytes': path.stat().st_size, 'load_ms': load_ms, 'python_heap_peak_during_predict_bytes': peak, 'labels': len(model.labels), 'dimensions': model.dimensions, 'channels': len(model.weights), 'n_values': list(model.n_values), 'platform': platform.platform(), 'python': platform.python_version(), 'cpu_count': os.cpu_count(), 'iterations_per_length': iterations, 'latency_by_characters': latency}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--mode', choices=['short', 'long', 'both'], default='both'); p.add_argument('--iterations', type=int, default=200); p.add_argument('--output', type=Path, default=Path('compact_hdc_v1/reports/detailed_benchmark.json')); a = p.parse_args()
    modes = ['short', 'long'] if a.mode == 'both' else [a.mode]
    reports = {mode: profile(Path(f'compact_hdc_v1/artifacts/lid235_{mode}.npz'), mode, a.iterations) for mode in modes}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(reports, indent=2), encoding='utf8'); print(json.dumps(reports, indent=2))

if __name__ == '__main__': main()

