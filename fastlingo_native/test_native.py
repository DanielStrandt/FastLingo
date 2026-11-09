"""Parity smoke tests for the compiled native engine.

Build the executable first, then run:
  python fastlingo_native/test_native.py
"""
from __future__ import annotations

import subprocess
import sys
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
]


def main() -> int:
    if not EXE.exists():
        print(f"missing {EXE}; build the native target first", file=sys.stderr)
        return 2
    from compact_hdc_v1.model import CompactIdentifier

    failures = []
    for mode in ("short", "long"):
        model_path = ROOT / "compact_hdc_v1" / "artifacts" / f"lid235_{mode}.npz"
        model = CompactIdentifier(model_path)
        expected = [model.labels.index(model.predict(text)["label"]) for text in CASES]
        payload = ("\n".join(CASES) + "\n").encode("utf-8")
        proc = subprocess.run(
            [str(EXE), str(ROOT / "fastlingo_native" / f"lid235_{mode}.bin"), "--stdin"],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        actual = [int(x) for x in proc.stdout.decode("ascii").splitlines() if x]
        if proc.returncode or actual != expected:
            failures.append((mode, expected, actual, proc.stderr.decode(errors="replace")))
        else:
            print(f"{mode}: {len(CASES)}/{len(CASES)} parity cases passed")
    if failures:
        for mode, expected, actual, err in failures:
            print(f"{mode}: expected={expected} actual={actual} stderr={err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

