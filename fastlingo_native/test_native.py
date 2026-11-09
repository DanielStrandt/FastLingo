"""Parity smoke tests for the compiled native engine.

Build the executable first, then run:
  python fastlingo_native/test_native.py
"""
from __future__ import annotations

import subprocess
import sys
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXE = ROOT / "fastlingo_native" / "fastlingo_native.exe"

CASES = [
    "This is an English sentence.",
    "Dies ist ein deutscher Satz.",
    "Ceci est une phrase française.",
    "これは日本語の文です。",
    "Русский текст для проверки.",
    "中文语言识别测试。",
    "ภาษาไทยสำหรับการทดสอบ",
    "Cafe\u0301 au lait — decomposed accent.",
    "Straße und STRASSE.",
    "KELVIN K and ligature ﬃ.",
    "Μάθηση ελληνικών: ὀδός.",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--executable", type=Path, default=EXE)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--native-dir", type=Path, default=ROOT / "fastlingo_native")
    args = p.parse_args()
    exe = args.executable
    model_dir = args.model_dir or ((ROOT / "artifacts") if (ROOT / "artifacts" / "lid235_short.npz").exists() else ROOT / "compact_hdc_v1" / "artifacts")
    package = "fastlingo" if (ROOT / "fastlingo").exists() else "compact_hdc_v1"
    if not exe.exists():
        print(f"missing {exe}; build the native target first", file=sys.stderr)
        return 2
    module = __import__(f"{package}.model", fromlist=["CompactIdentifier"])
    CompactIdentifier = module.CompactIdentifier

    # Expand the hand-written corpus into deterministic stress cases so parity
    # covers many token lengths and repeated normalization paths, not only 11 labels.
    cases = CASES + [f"{text} sample-{i}" for i in range(100) for text in CASES]
    failures = []
    for mode in ("short", "long"):
        model_path = model_dir / f"lid235_{mode}.npz"
        model = CompactIdentifier(model_path)
        expected = [model.labels.index(model.predict(text)["label"]) for text in cases]
        payload = ("\n".join(cases) + "\n").encode("utf-8")
        proc = subprocess.run(
            [str(exe), str(args.native_dir / f"lid235_{mode}.bin"), "--stdin"],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        actual = [int(x) for x in proc.stdout.decode("ascii").splitlines() if x]
        if proc.returncode or actual != expected:
            failures.append((mode, expected, actual, proc.stderr.decode(errors="replace")))
        else:
            print(f"{mode}: {len(cases)}/{len(cases)} parity cases passed")
    if failures:
        for mode, expected, actual, err in failures:
            print(f"{mode}: expected={expected} actual={actual} stderr={err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

