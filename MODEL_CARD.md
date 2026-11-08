# FastLingo model card

## Model summary

FastLingo is a two-artifact HDC language identifier covering the 235 labels in WiLI-2018. The short and long artifacts are trained and deployed independently; applications select the artifact appropriate for the expected text length.

## Intended use

- Offline language identification
- Routing text to language-specific processing
- Lightweight edge and desktop applications
- Privacy-sensitive preprocessing where sending text to a remote model is undesirable

## Out of scope

- Identification of languages outside the 235-label set
- Person, nationality, ethnicity, or location inference
- High-stakes decisions
- Sole evidence for moderation, access control, or legal decisions

## Evaluation

The long model is evaluated on the complete WiLI test split. The short model uses a deterministic mixed WiLI/Tatoeba holdout. Exact counts, scores, and benchmark methodology are in [RESULTS.md](RESULTS.md) and `reports/`.

## Limitations

Performance can fall on very short strings, names, code-switching, transliteration, noisy text, domain shifts, and closely related languages. Applications should expose confidence and abstain when the margin is insufficient.

## Data and licensing

See [DATA_LICENSES.md](DATA_LICENSES.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The source-code license does not automatically relicense the training data or model artifacts.
