$ErrorActionPreference = 'Stop'

$modelDir = if (Test-Path artifacts) { 'artifacts' } else { 'compact_hdc_v1/artifacts' }
python fastlingo_native/export_model.py "$modelDir/lid235_short.npz" fastlingo_native/lid235_short.bin
python fastlingo_native/export_model.py "$modelDir/lid235_long.npz" fastlingo_native/lid235_long.bin
cmake -S fastlingo_native -B fastlingo_native/build -DCMAKE_BUILD_TYPE=Release
cmake --build fastlingo_native/build --config Release

