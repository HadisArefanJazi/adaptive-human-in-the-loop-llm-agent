"""reproduce the benchmark across training seeds without selecting a best run."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path as path_type
from statistics import mean, stdev

from adaptive_hitl_agent.experiment import experiment_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 3, 5, 7, 11, 13, 17, 19])
    parser.add_argument(
        "--reference-seed", type=int, default=7,
        help="seed whose full metrics are also exported (default: 7, if included).",
    )
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--output-dir", type=path_type, default=path_type("results"))
    parser.add_argument("--artifacts-dir", type=path_type, default=path_type("artifacts/benchmark"))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    if args.episodes < 1:
        parser.error("episodes must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    metadata = {}
    for seed in args.seeds:
        artifact_dir = args.artifacts_dir / f"seed{seed}"
        result = run_experiment(
            experiment_config(training_episodes=args.episodes, seed=seed),
            output_dir=artifact_dir,
        )
        runs[str(seed)] = {name: metrics.as_dict() for name, metrics in result.metrics.items()}
        metadata = result.metadata
        if seed == args.reference_seed:
            shutil.copyfile(artifact_dir / "metrics.json", args.output_dir / f"benchmark_seed{seed}.json")
        print(f"seed {seed}: system success={result.metrics['adaptive_rl'].system_success:.1%}")

    first = runs[str(args.seeds[0])]
    summary = {}
    for policy, metrics in first.items():
        summary[policy] = {}
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                values = [run[policy][metric] for run in runs.values()]
                summary[policy][metric] = {
                    "mean": mean(values),
                    "sample_std": stdev(values) if len(values) > 1 else 0.0,
                }
    payload = {
        "seeds": args.seeds,
        "training_episodes": args.episodes,
        "reference_seed": args.reference_seed if args.reference_seed in args.seeds else None,
        "metadata": metadata,
        "summary": summary,
        "runs": runs,
    }
    path = args.output_dir / "benchmark_multiseed.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"results written to {path}")


if __name__ == "__main__":
    main()
