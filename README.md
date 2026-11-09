# FastLingo

FastLingo is a compact, offline language identifier for 235 written languages, built with sparse hyperdimensional computing (HDC).

It ships two independent models:

- **Short model** for sentences and messages
- **Long model** for paragraphs and longer context

Each model is standalone, CPU-friendly, and smaller than 1 MB.

## Verified release results

| Model | Languages | Artifact | Held-out accuracy | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| Short | 235 | 623.6 kB | 86.41% | 1.72 ms | 2.14 ms |
| Long | 235 | 574.8 kB | 92.10% | 2.19 ms | 2.59 ms |

The short score is measured on a deterministic mixed WiLI/Tatoeba holdout. The long score is measured on the complete 117,500-example WiLI test split. Latency is measured with Python on Windows 11 using 100 iterations per input length; it is a reproducible implementation benchmark, not a universal edge-device guarantee.

## Independent external evaluation

To test transfer beyond the training sources, we evaluated each model on a separate public benchmark and compared it with fastText on the same examples. Scores use the shared 166-language ISO label intersection and a balanced, deterministic sample of up to 100 examples per label.

| External benchmark | FastLingo model | FastLingo accuracy | fastText accuracy | FastLingo macro-F1 | fastText macro-F1 |
|---|---|---:|---:|---:|---:|
| FLORES-200 devtest (short) | Short | **88.53%** | 81.13% | **89.61%** | 79.46% |
| UDHR-LID test, texts ≥240 chars (long) | Long | **88.91%** | 75.59% | **90.10%** | 72.62% |

These are external evaluation results, not training-set scores. See [the raw report](reports/external_comparison.json) and [the reproducible evaluator](fastlingo/evaluate_external.py).

## Matched fastText comparison

On the WiLI test intersection, FastLingo-long reached **93.26%** accuracy versus **72.21%** for the official lid.176.ftz model, while using a 574.8 kB artifact versus 938.0 kB. The fastText Python binding was much faster because it executes optimized native C++; the current FastLingo benchmark is Python/NumPy, so this is not a native-to-native speed claim. See [RESULTS.md](RESULTS.md) for the full protocol and length sweep.

## Accelerated inference path

For low-latency deployment, install the optional compiled numeric path:

    python -m pip install -r requirements-accelerated.txt
    python -c "from fastlingo.accelerated import AcceleratedIdentifier; print(AcceleratedIdentifier('artifacts/lid235_long.npz').predict('This is an English sentence.'))"

On the recorded Windows 11 host, the accelerated short model measured **0.439 ms p50 / 0.520 ms p95**, and the accelerated long model measured **0.475 ms p50 / 0.640 ms p95** for representative inputs, versus roughly 3 ms for the reference Python path. It agreed with the reference model on **15,456 of 15,457** multilingual external examples (99.9935%). This is a Python+Numba optimization and is still not a native-to-native comparison with fastText. A dependency-free C++17 execution layer is included under [fastlingo_native/](fastlingo_native/README.md); native latency is intentionally not claimed until the binary is measured on the target device.

## Why FastLingo

- 235-language output space
- No neural-network runtime
- Offline inference
- Sub-megabyte model artifacts
- Deterministic training and evaluation
- Separate short- and long-context operating points
- Transparent reports and legal dataset notices

FastLingo is designed for applications where a small, private, low-latency language signal is more useful than a large general-purpose model.

## Quick start

From the repository root:

```powershell
python -m pip install -r requirements.txt
python -m fastlingo.train --mode long
python -c "from fastlingo.model import CompactIdentifier; print(CompactIdentifier('artifacts/lid235_long.npz').predict('This is an English sentence.'))"
```

The source repository intentionally omits binary model artifacts. Train them locally with the reproducible commands below, or distribute a separately reviewed release asset with its own checksum and data-rights notice.

## Reproducing the models

Training data is not included. Obtain WiLI-2018 and the prepared Tatoeba mixture independently, following [DATA_LICENSES.md](DATA_LICENSES.md).

```powershell
python -m fastlingo.train --mode short
python -m fastlingo.train --mode long
python -m fastlingo.evaluate --mode short
python -m fastlingo.evaluate --mode long
python -m fastlingo.benchmark --mode both --iterations 100
# Optional baseline comparison (install requirements-benchmark.txt first)
python -m fastlingo.benchmark_fasttext --fasttext-model path/to/lid.176.ftz
```

Generated artifacts and reports are written under `artifacts/` and `reports/`.

## Scope and limitations

FastLingo recognizes the 235 labels present in WiLI-2018. It does not identify arbitrary languages outside that label set, and the model should not be used to infer a user's identity, ethnicity, location, or nationality. Very short, noisy, code-switched, transliterated, and domain-specific text can be difficult. Use the confidence margin and application-level abstention policy when incorrect predictions are costly.

## Documentation

- [Model card](MODEL_CARD.md)
- [Results and evaluation protocol](RESULTS.md)
- [Dataset and license notices](DATA_LICENSES.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Citation](CITATION.cff)
- Raw benchmark profile: `reports/detailed_benchmark.json`
- Matched fastText comparison: [RESULTS.md](RESULTS.md#matched-fasttext-comparison)

## License

The source code is MIT-licensed. Training data and model artifacts are subject to their own terms; see [DATA_LICENSES.md](DATA_LICENSES.md). This repository is not legal advice.
