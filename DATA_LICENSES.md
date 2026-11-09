# Dataset and data-derived artifact notices

FastLingo's source code is MIT-licensed. The training corpora are not included in this repository and remain subject to their own terms. Keep the exact download URL, version, date, checksum, preprocessing manifest, and attribution text with every reproducible training run.

## WiLI-2018

WiLI-2018 is the written-language identification benchmark by Martin Thoma. Use the canonical Zenodo record and paper when obtaining it:

- Dataset: https://doi.org/10.5281/zenodo.841984
- Paper: https://arxiv.org/abs/1801.07779
- Dataset metadata identifies an Open Database License (ODbL) family license.

Do not assume that MIT licensing of FastLingo's code relicenses WiLI. If redistributing a copy, modified database, or database-derived distribution, preserve the applicable attribution and notices and review the ODbL obligations for the exact dataset version.

## Tatoeba

FastLingo uses monolingual text obtained through the Tatoeba/OPUS ecosystem. Tatoeba's textual sentences are generally released under CC BY 2.0 FR, with attribution required:

- Tatoeba: https://tatoeba.org
- Tatoeba usage guidance: https://en.wiki.tatoeba.org/articles/show/using-the-tatoeba-corpus
- Tatoeba terms: https://tatoeba.org/en/terms_of_use
- CC BY 2.0 FR: https://creativecommons.org/licenses/by/2.0/fr/

Attribute Tatoeba and preserve contributor/provenance information where applicable. Do not redistribute raw sentence files through this repository unless you have verified the exact source terms.

## Model artifacts

The legal status of trained parameters can depend on the jurisdiction, exact data subset, preprocessing, and distribution format. This repository therefore does not represent model artifacts as automatically MIT-licensed. Before commercial or broad redistribution, review the applicable dataset and database-rights questions with qualified counsel.

This notice is compliance guidance, not legal advice.


## External evaluation-only datasets

These datasets were used only for independent evaluation and are not training inputs for FastLingo.

### FLORES-200

The official FLORES-200 release is licensed CC BY-SA 4.0. Preserve attribution and ShareAlike obligations for any redistribution of the evaluation data or adaptations. Source: https://github.com/facebookresearch/flores and https://creativecommons.org/licenses/by-sa/4.0/.

### UDHR-LID

The UDHR-LID packaging, metadata, and annotations are released under CC0 1.0 according to its dataset card. Cite the UDHR source and the GlotLID/UDHR-LID resource when using the benchmark: https://huggingface.co/datasets/cis-lmu/udhr-lid.
