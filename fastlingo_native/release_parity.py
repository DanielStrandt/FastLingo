"""Full-artifact native parity check for release qualification.

This is intentionally separate from the fast synthetic CI fixture. It compares
the real 235-language artifacts on deterministic samples from FLORES-200 and
UDHR-LID, using the same short/long routing split as the external evaluation.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run_native(exe: Path, model: Path, texts: list[str]) -> list[int]:
    payload = ("\n".join(t.replace("\r", " ").replace("\n", " ") for t in texts) + "\n").encode("utf-8")
    p = subprocess.run([str(exe), str(model), "--stdin"], input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace"))
    return [int(x) for x in p.stdout.decode("ascii").splitlines() if x]


def check(mode: str, texts: list[str], package, model_path: Path, native_model: Path, exe: Path) -> tuple[int, int]:
    model = package.CompactIdentifier(model_path)
    expected = [model.labels.index(model.predict(t)["label"]) for t in texts]
    actual = run_native(exe, native_model, texts)
    matches = sum(a == b for a, b in zip(actual, expected))
    print(f"{mode}: {matches}/{len(texts)} exact label matches")
    shown = 0
    for i, (a, b) in enumerate(zip(actual, expected)):
        if a != b and shown < 5:
            print(f"  mismatch[{i}]: native={a} python={b} text={texts[i][:120].encode('unicode_escape').decode('ascii')}")
            shown += 1
    return matches, len(texts)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--executable", type=Path, default=ROOT / "fastlingo_native" / "fastlingo_native.exe")
    p.add_argument("--model-dir", type=Path, default=ROOT / "compact_hdc_v1" / "artifacts")
    p.add_argument("--flores-dir", type=Path, default=ROOT / "data" / "raw" / "flores200_dataset")
    p.add_argument("--udhr-parquet", type=Path, default=ROOT / "data" / "raw" / "udhr_test.parquet")
    p.add_argument("--per-label", type=int, default=100)
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    try:
        from compact_hdc_v1 import model as package
    except ModuleNotFoundError:
        from fastlingo import model as package

    flores_files = sorted((args.flores_dir / "devtest").glob("*.devtest"))
    if not flores_files:
        raise SystemExit("FLORES devtest file not found")
    flores = []
    for flores_file in flores_files:
        flores.extend(flores_file.read_text(encoding="utf-8").splitlines()[:args.per_label])
    udhr = pd.read_parquet(args.udhr_parquet)
    udhr = udhr[udhr["sentence"].str.len() >= 240]["sentence"].tolist()[: args.per_label * 166]
    total = 0; matches = 0
    for mode, texts in (("short", flores), ("long", udhr)):
        m, n = check(mode, texts, package, args.model_dir / f"lid235_{mode}.npz", ROOT / "fastlingo_native" / f"lid235_{mode}.bin", args.executable)
        matches += m; total += n
    print(f"overall: {matches}/{total} exact label matches")
    return 0 if matches == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

