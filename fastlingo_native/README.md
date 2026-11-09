# FastLingo native execution layer

This directory contains a dependency-free C++17 inference engine and model exporter.
It is designed to preserve the compact HDC model while removing Python/NumPy overhead
from production inference.

## Build

On Windows, install the **Desktop development with C++** workload (including the MSVC
toolset and Windows SDK), then run from the repository root:

```powershell
python fastlingo_native/export_model.py artifacts/lid235_short.npz fastlingo_native/lid235_short.bin
python fastlingo_native/export_model.py artifacts/lid235_long.npz fastlingo_native/lid235_long.bin
cmake -S fastlingo_native -B fastlingo_native/build -DCMAKE_BUILD_TYPE=Release
cmake --build fastlingo_native/build --config Release
```

The executable accepts a model, UTF-8 text, and an optional iteration count:

```powershell
fastlingo_native/build/Release/fastlingo_native.exe fastlingo_native/lid235_long.bin "A representative sentence." 10000
```

The exporter writes a versioned flat binary containing the model metadata, weights,
bias, quantized prototypes, and labels. The loader validates the header and dimensions
before scoring.

## Current fallback benchmark

The checked-in Python/Numba execution path is the reference-compatible fallback when a
C++ toolchain is unavailable. On the development machine it measures approximately
0.45 ms p50 per prediction after JIT warm-up. The compiled AVX2 C++ engine measured
0.228 ms for the short representative input and 0.230 ms for the long representative
input over 5,000 warmed iterations. These are host-specific measurements excluding
process startup and model loading; benchmark the target device before deployment.

The native implementation keeps the same model math and label set. On Windows it uses
NFC normalization plus invariant lowercase mapping with explicit handling for common
case-fold expansions (such as ß and ligatures). The parity suite covers decomposed
accents, case-fold-sensitive text, combining marks, and multiple non-Latin scripts.
Run the suite on the target platform before release; non-Windows builds still require
an equivalent Unicode normalization provider for full parity.


## Correctness and comparison checks

After building the executable, run the UTF-8 parity suite:

```powershell
python fastlingo_native/test_native.py
```

The suite exercises both models on 1,111 deterministic multilingual and normalization-sensitive UTF-8 cases and compares native label indices with the Python reference. The current development run passed 11/11 cases for each model.

For a reproducible warm-speed comparison with the official fastText binding:

```powershell
python fastlingo_native/benchmark_native_vs_fasttext.py --mode short --iterations 1000
python fastlingo_native/benchmark_native_vs_fasttext.py --mode long --iterations 1000
```

On the development host, the internal native timer measured approximately 0.228 ms (short) and 0.230 ms (long), while fastText measured approximately 0.0081 ms and 0.0118 ms respectively. The native stdin figure includes pipe overhead and is not used as the warm inference claim. These measurements are host-specific and should be rerun on the deployment device.


For CI, `create_fixture.py` creates deterministic non-shipped artifacts and the Windows native job builds, exports, and runs the parity suite on every push and pull request.

## Release qualification

The synthetic CI fixture is intentionally small and fast. For release qualification, run the real-artifact check:

```powershell
python fastlingo_native/release_parity.py --per-label 100
```

This compares native and Python predictions on deterministic FLORES-200 and UDHR-LID samples using the actual 235-language, 8-channel, 768-dimensional models. It is a gate, not a synthetic smoke test. The stored development report (`reports/native_release_parity.json`) records the model hashes, environment, sampling protocol, and mismatch examples. After matching Python's removal of Unicode format characters (including zero-width spaces), the latest balanced intersection run matched 2,005/2,005 labels. The stored report records this passing run and its hashes. Native C++ should still be requalified whenever the model, compiler, or target platform changes.

The flat native export is intentionally uncompressed and is approximately 1.45 MB per model. The original compressed NPZ artifacts remain approximately 575–624 kB. Therefore the native binary currently prioritizes execution simplicity over the sub-1 MB artifact target.
