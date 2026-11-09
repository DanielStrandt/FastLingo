# FastLingo native execution layer

This directory contains a dependency-free C++17 inference engine and model exporter.
It is designed to preserve the compact HDC model while removing Python/NumPy overhead
from production inference.

## Build

On Windows, install the **Desktop development with C++** workload (including the MSVC
toolset and Windows SDK), then run from the repository root:

```powershell
python fastlingo_native/export_model.py compact_hdc_v1/artifacts/lid235_short.npz fastlingo_native/lid235_short.bin
python fastlingo_native/export_model.py compact_hdc_v1/artifacts/lid235_long.npz fastlingo_native/lid235_long.bin
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
0.181 ms for the short representative input and 0.186 ms for the long representative
input over 10,000 warmed iterations. These are host-specific measurements excluding
process startup and model loading; benchmark the target device before deployment.

The native implementation intentionally keeps the same model math and label set. Its
Unicode normalization is conservative; use the Python/Numba path for full Unicode NFC
and case-folding requirements until native-vs-reference agreement has been validated
on the target corpus.

