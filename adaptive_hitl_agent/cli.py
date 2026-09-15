from __future__ import annotations

import argparse

from .evaluation import format_metrics
from .experiment import experiment_config, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive-hitl",
        description="train and compare a cost-aware human-in-the-loop routing agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    experiment = subparsers.add_parser(
        "experiment",
        help="train the dqn router and evaluate it against comparison policies.",
    )
    experiment.add_argument("--episodes", type=int, default=600)
    experiment.add_argument("--seed", type=int, default=7)
    experiment.add_argument("--output-dir", default="artifacts/latest")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "experiment":
        try:
            config = experiment_config(training_episodes=args.episodes, seed=args.seed)
        except ValueError as error:
            parser.error(str(error))
        result = run_experiment(config=config, output_dir=args.output_dir)
        print(format_metrics(result.metrics))
        print(f"\nartifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()
