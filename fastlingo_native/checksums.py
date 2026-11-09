"""Print SHA-256 checksums for release artifacts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("paths", nargs="+", type=Path); p.add_argument("--output", type=Path)
    a = p.parse_args(); manifest = {str(path): {"sha256": sha256(path), "bytes": path.stat().st_size} for path in a.paths}
    text = json.dumps(manifest, indent=2) + "\n"
    if a.output: a.output.write_text(text, encoding="utf-8")
    print(text, end="")

if __name__ == "__main__": main()

