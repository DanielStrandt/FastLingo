"""Full-artifact native parity check for release qualification.

This is intentionally separate from the fast synthetic CI fixture. It compares
the real 235-language artifacts on deterministic samples from FLORES-200 and
UDHR-LID, using the same short/long routing split as the external evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
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


def check(mode: str, rows: list[tuple[str, str]], package, model_path: Path, native_model: Path, exe: Path) -> tuple[int, int, list[dict]]:
    texts = [text for text, _ in rows]
    model = package.CompactIdentifier(model_path)
    expected = [model.labels.index(model.predict(t)["label"]) for t in texts]
    actual = run_native(exe, native_model, texts)
    if len(actual) != len(expected):
        raise RuntimeError(f"native output count {len(actual)} != reference count {len(expected)}")
    matches = sum(a == b for a, b in zip(actual, expected))
    print(f"{mode}: {matches}/{len(texts)} exact label matches")
    mismatches = []
    shown = 0
    for i, (a, b) in enumerate(zip(actual, expected)):
        if a != b and shown < 5:
            print(f"  mismatch[{i}]: native={a} python={b} text={texts[i][:120].encode('unicode_escape').decode('ascii')}")
            mismatches.append({"index": i, "native_index": a, "python_index": b, "text": texts[i]})
            shown += 1
    return matches, len(texts), mismatches


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--executable", type=Path, default=ROOT / "fastlingo_native" / "fastlingo_native.exe")
    default_model_dir = ROOT / "artifacts" if (ROOT / "artifacts" / "lid235_short.npz").exists() else ROOT / "compact_hdc_v1" / "artifacts"
    p.add_argument("--model-dir", type=Path, default=default_model_dir)
    p.add_argument("--flores-dir", type=Path, default=ROOT / "data" / "raw" / "flores200_dataset")
    p.add_argument("--udhr-parquet", type=Path, default=ROOT / "data" / "raw" / "udhr_test.parquet")
    p.add_argument("--fasttext-model", type=Path, default=ROOT / "fasttext_lid.176.ftz")
    p.add_argument("--per-label", type=int, default=100)
    p.add_argument("--report", type=Path, default=ROOT / "reports" / "native_release_parity.json")
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
    common_labels = None
    if args.fasttext_model.exists():
        try:
            import fasttext, pycountry
            raw = {x.removeprefix("__label__") for x in fasttext.load_model(str(args.fasttext_model)).get_labels()}
            common_labels = {item.alpha_3 if (item := pycountry.languages.get(alpha_2=code)) else code for code in raw}
            common_labels.update({"heb", "ind", "yid", "jav", "ron", "srp"})
        except Exception:
            common_labels = None
    flores_rows = []
    for flores_file in flores_files:
        label = flores_file.name.split("_")[0]
        for line in flores_file.read_text(encoding="utf-8").splitlines():
            if line.strip(): flores_rows.append((line.strip(), label))
    udhr = pd.read_parquet(args.udhr_parquet)
    udhr_rows = [(str(row["sentence"]), str(row["iso639-3"])) for _, row in udhr.iterrows() if len(str(row["sentence"])) >= 240]
    total = 0; matches = 0; report = {"sampling": {"per_label": args.per_label, "long_min_chars": 240}, "environment": {"platform": platform.platform(), "python": platform.python_version()}, "datasets": {}}
    for mode, rows in (("short", flores_rows), ("long", udhr_rows)):
        model_path = args.model_dir / f"lid235_{mode}.npz"; model = package.CompactIdentifier(model_path); seen: dict[str, int] = {}; selected = []
        for text, label in rows:
            if label not in model.labels or (common_labels is not None and label not in common_labels) or seen.get(label, 0) >= args.per_label: continue
            selected.append((text, label)); seen[label] = seen.get(label, 0) + 1
        m, n, mismatches = check(mode, selected, package, model_path, ROOT / "fastlingo_native" / f"lid235_{mode}.bin", args.executable)
        matches += m; total += n
        report["datasets"][mode] = {"examples": n, "matches": m, "accuracy": m / max(1, n), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "mismatches": mismatches}
    report["overall"] = {"examples": total, "matches": matches, "accuracy": matches / max(1, total)}
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if matches == total else 1



if __name__ == "__main__":
    raise SystemExit(main())

