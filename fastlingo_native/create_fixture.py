"""Create tiny deterministic CI fixtures without shipping model artifacts."""
from pathlib import Path
import numpy as np

def main() -> None:
    out = Path("artifacts"); out.mkdir(exist_ok=True)
    rng = np.random.default_rng(235)
    labels = np.array(["deu", "eng", "fra"])
    names = np.array(["German", "English", "French"])
    for mode in ("short", "long"):
        np.savez(
            out / f"lid235_{mode}.npz",
            labels=labels, label_names=names,
            prototypes=rng.integers(-8, 9, size=(4, 3, 64), dtype=np.int8),
            channel_weights=np.ones(4, dtype=np.float32) / 4,
            class_bias=np.zeros(3, dtype=np.float32),
            dimensions=np.int64(64), sparsity=np.int64(2),
            n_values=np.array([1, 2], dtype=np.int64), seed=np.int64(20261108),
        )

if __name__ == "__main__":
    main()

