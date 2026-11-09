"""Reproducible native-vs-fastText timing comparison.

This reports warm prediction time separately from process/model startup. It does
not mix the two measurements, because loading a model is not inference latency.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
import re
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("short", "long"), default="long")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--fasttext-model", type=Path, default=Path("fasttext_lid.176.ftz"))
    p.add_argument("--native-exe", type=Path, default=Path("fastlingo_native/fastlingo_native.exe"))
    p.add_argument("--native-model", type=Path)
    args = p.parse_args()
    text = (
        "This is a short sentence for edge inference."
        if args.mode == "short"
        else "This is a representative language identification sentence used for edge inference."
    )
    native_model = args.native_model or Path(f"fastlingo_native/lid235_{args.mode}.bin")

    from fasttext import load_model

    ft = load_model(str(args.fasttext_model))
    ft.predict(text)
    ft_times = []
    for _ in range(args.iterations):
        t0 = time.perf_counter_ns()
        ft.predict(text)
        ft_times.append((time.perf_counter_ns() - t0) / 1e6)

    payload = ((text + "\n") * args.iterations).encode("utf-8")
    t0 = time.perf_counter_ns()
    proc = subprocess.run(
        [str(args.native_exe), str(native_model), "--stdin"],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    native_batch_ms = (time.perf_counter_ns() - t0) / 1e6
    if proc.returncode:
        raise SystemExit(proc.stderr.decode(errors="replace"))
    native_warm = native_batch_ms / args.iterations
    single = subprocess.run(
        [str(args.native_exe), str(native_model), text, str(args.iterations)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if single.returncode:
        raise SystemExit(single.stderr)
    match = re.search(r"avg_ms=([0-9.]+)", single.stdout)
    if not match:
        raise SystemExit(f"unexpected native timing output: {single.stdout}")
    out = {
        "mode": args.mode,
        "iterations": args.iterations,
        "fasttext_warm_p50_ms": statistics.median(ft_times),
        "fasttext_warm_mean_ms": statistics.fmean(ft_times),
        "native_stdin_batch_mean_ms": native_warm,
        "native_warm_internal_avg_ms": float(match.group(1)),
        "native_process_plus_batch_ms": native_batch_ms,
        "note": "Native stdin timing includes pipe/batch overhead; single-call native timing is reported by the executable's iteration mode.",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

