from __future__ import annotations

import argparse

from .experiment import ExperimentConfig, format_metrics, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-hitl",
        description="Train and compare a resource-aware human-in-the-loop agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    experiment = subparsers.add_parser(
        "experiment",
        help="Train the DQN router and evaluate it against all baselines.",
    )
    experiment.add_argument("--episodes", type=int, default=600)
    experiment.add_argument("--seed", type=int, default=7)
    experiment.add_argument("--output-dir", default="artifacts/latest")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "experiment":
        config = ExperimentConfig(training_episodes=args.episodes, seed=args.seed)
        result = run_experiment(config=config, output_dir=args.output_dir)
        print(format_metrics(result.metrics))
        print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
