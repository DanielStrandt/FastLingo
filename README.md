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
