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


## Matched fastText comparison

Using the official `lid.176.ftz` model (938,013 bytes), an explicit ISO-639-1 to ISO-639-3 mapping, and the WiLI test split restricted to 166 semantically matched labels (83,000 examples):

| Model | Accuracy | Artifact | p50 prediction | p95 prediction |
|---|---:|---:|---:|---:|
| FastLingo long | 93.26% | 574.8 kB | 3.15 ms | 3.66 ms |
| fastText `lid.176.ftz` | 72.21% | 938.0 kB | 0.009 ms | 0.009 ms |

This is a label-intersection comparison, not a claim over fastText's unsupported languages. The speed figures are specifically Python fastText binding versus Python/NumPy FastLingo; they are not a native algorithm comparison. A length sweep shows the gap narrowing from about 269x at 16 characters to about 29x at 2,048 characters, because FastLingo's feature work is mostly fixed-cost while fastText's native path scales with input. FastLingo's demonstrated advantages here are output coverage, accuracy on this WiLI slice, and smaller artifact size. Reproduce it with `requirements-benchmark.txt` and `python -m fastlingo.benchmark_fasttext --fasttext-model path/to/lid.176.ftz`. Raw mapping and measurements are in `reports/fasttext_comparison.json` and `reports/fasttext_length_sweep.json`.


## Clean external evaluation

To test whether the advantage transfers beyond WiLI/Tatoeba-derived data, FastLingo was evaluated on two public external datasets with the same examples sent to fastText. Labels were normalized from ISO-639-1 to ISO-639-3 and scores were computed only on the 166-label semantic intersection. Sampling is deterministic and capped at 100 examples per label for balanced runtime; the long set additionally requires at least 240 characters per example.

| External dataset | FastLingo artifact | Examples | FastLingo accuracy | fastText accuracy |
|---|---|---:|---:|---:|
| FLORES-200 devtest | Short | 9,800 | 88.53% | 81.13% |
| UDHR-LID test, text ≥240 chars | Long | 2,147 | 88.91% | 75.59% |

FLORES-200 is a multilingual translation benchmark; UDHR-LID is a separate public human-rights text benchmark. These are external evaluations, not training data. Reproduce them with `python -m fastlingo.evaluate_external` after installing `requirements-benchmark.txt` and downloading the public datasets. Raw results are in `reports/external_comparison.json`.
