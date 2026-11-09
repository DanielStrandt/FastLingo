# Release and production checklist

The Git repository does **not** ship trained model binaries. Users must either:

1. obtain the training data under its applicable terms and train the short/long
   artifacts locally, or
2. obtain a separately published release bundle that includes the model files,
   checksums, provenance, and the applicable data-rights notices.

Never describe a model binary as MIT-licensed solely because the source code is MIT-licensed.

## Release bundle requirements

For every distributed model artifact, publish:

- the exact file name and model mode (`short` or `long`);
- SHA-256 checksum;
- training commit or source revision;
- dataset names, versions, download URLs, and acquisition dates;
- preprocessing and tokenizer configuration;
- model dimensions, channels, seed, and label manifest;
- `DATA_LICENSES.md` and `THIRD_PARTY_NOTICES.md`;
- the evaluation and native-parity reports used to qualify the artifact.

Generate checksums on Windows with:

```powershell
Get-FileHash artifacts/lid235_short.npz -Algorithm SHA256
Get-FileHash artifacts/lid235_long.npz -Algorithm SHA256
```

Generate checksums on Unix-like systems with:

```bash
sha256sum artifacts/lid235_short.npz artifacts/lid235_long.npz
```

Record the output in the release notes or a manifest shipped beside the binaries.

## Production validation

Before production-critical deployment, rerun the following on the target hardware and
runtime:

1. Python/native parity on the real artifacts:
   `python fastlingo_native/release_parity.py --per-label 100`.
2. Native UTF-8 and normalization tests:
   `python fastlingo_native/test_native.py`.
3. Full application routing tests, including short/long boundary cases, empty input,
   malformed UTF-8 handling, code switching, and abstention thresholds.
4. Accuracy and macro-F1 on the intended domain, plus a representative multilingual
   stress set—not only the public benchmarks.
5. p50/p95/p99 latency, throughput, peak resident memory, cold-start time, and binary
   loading time on the actual CPU/OS/compiler configuration.
6. Artifact integrity verification using the published SHA-256 values.
7. A rollback path to the Python/reference implementation or a previously qualified
   native binary.

The published benchmark numbers are development-host measurements and are not service
level objectives for another device.

