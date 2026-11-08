# Model artifacts

Binary model artifacts are intentionally not committed to the source branch. Generate them with:

    python -m fastlingo.train --mode short
    python -m fastlingo.train --mode long

Before publishing a binary release asset, record its SHA-256 checksum and preserve the applicable dataset/data-derived artifact notice from DATA_LICENSES.md.
