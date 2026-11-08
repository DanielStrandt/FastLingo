# Release checklist

Before publishing a binary FastLingo release:

- [ ] Pin the exact WiLI-2018 and Tatoeba/OPUS sources, dates, and checksums.
- [ ] Re-run training, evaluation, and benchmark commands from a clean checkout.
- [ ] Confirm label order and artifact compatibility metadata.
- [ ] Generate SHA-256 checksums for each model artifact.
- [ ] Attach only artifacts whose data-derived distribution has been reviewed.
- [ ] Publish the matching reports and third-party notices with the release.
- [ ] Report unsupported languages rather than mapping them to a supported label.
- [ ] Repeat latency and memory measurements on the intended edge hardware.
