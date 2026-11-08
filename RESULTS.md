# Results

## Accuracy

| Artifact | Evaluation set | Examples | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| Short | Deterministic mixed WiLI/Tatoeba holdout | 8,680 | 86.41% | 86.83% |
| Long | WiLI-2018 official test split | 117,500 | 92.10% | 92.54% |

The two scores answer different questions and must not be compared as if they were the same benchmark.

## Latency

Representative warm prediction measurements on Windows 11, Python 3.12, 16 logical CPUs, 100 iterations per length:

| Artifact | Input | Size | Load | p50 | p95 | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Short | 16 characters | 623.6 kB | 13.7 ms | 1.72 ms | 2.14 ms | 549/s |
| Short | 240 characters | 623.6 kB | 13.7 ms | 2.59 ms | 3.14 ms | 377/s |
| Long | 128 characters | 574.8 kB | 7.7 ms | 2.19 ms | 2.59 ms | 450/s |
| Long | 2,048 characters | 574.8 kB | 7.7 ms | 2.92 ms | 3.54 ms | 334/s |

Raw multi-length measurements are in `reports/detailed_benchmark.json`. These host-Python measurements should be repeated on the intended edge CPU before making device-specific claims.

## Reproduction

```powershell
python -m fastlingo.evaluate --mode short
python -m fastlingo.evaluate --mode long
python -m fastlingo.benchmark --mode both --iterations 100
```
