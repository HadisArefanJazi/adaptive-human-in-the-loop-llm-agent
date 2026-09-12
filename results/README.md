# Benchmark artifacts

The JSON files here are recorded outputs, not configuration or training inputs.
`benchmark_seed7.json` contains full metrics and training metadata for the reference run.
`benchmark_multiseed.json` contains all eight runs and their mean/sample standard deviation.
The original recorded results are retained; do not hand-edit metrics to match a new run.

Reproduce without overwriting the recorded results:

```bash
python scripts/benchmark.py --output-dir artifacts/verification/results --artifacts-dir artifacts/verification/runs
```

The script exports full metrics for `--reference-seed` (default 7) if that seed is included in `--seeds`. Every requested seed contributes to the aggregate. Trace files and inference checkpoints are stored under the artifacts directory. Use a fresh output directory for a different seed set to avoid retaining old exports.

New runs also record package version, operating system, machine architecture, and training device alongside Python/PyTorch versions and dataset hashes. Older recorded files may omit these additive metadata fields. Minor floating-point differences across environments are expected. Training seeds measure variation on this fixed synthetic test set, not population uncertainty. See the main README for metric definitions and simulation limitations.
