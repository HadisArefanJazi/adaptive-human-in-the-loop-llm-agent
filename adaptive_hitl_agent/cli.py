from __future__ import annotations

import argparse
from pathlib import Path as path_type

from .dqn import dqn_policy
from .environment import assistance_environment
from .evaluation import format_metrics
from .experiment import experiment_config, run_experiment
from .llm import rule_based_language_model
from .retrieval import bm25_retriever
from .tools import safe_calculator
from .types import task_record


class terminal_human_reviewer:
    def answer(self, task: task_record) -> str:
        return input("human answer: ").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adaptive-hitl")
    commands = parser.add_subparsers(dest="command", required=True)

    experiment = commands.add_parser("experiment")
    experiment.add_argument("--episodes", type=int, default=3000)
    experiment.add_argument("--seed", type=int, default=7)
    experiment.add_argument("--output-dir", default="artifacts/latest")

    ask = commands.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument("--policy", default="artifacts/latest/dqn_policy.pt")

    return parser


def ask_question(question: str, policy_path: str) -> None:
    policy = dqn_policy.load(path_type(policy_path))

    task = task_record(
        "interactive",
        question,
        ("unknown",),
        "test",
        "interactive",
    )

    environment = assistance_environment(
        task,
        bm25_retriever.from_package_data(),
        rule_based_language_model(),
        safe_calculator(),
        terminal_human_reviewer(),
    )

    observation = environment.observe()

    while True:
        action = policy.select_action(
            observation,
            environment.available_actions(),
        )

        print("action:", action.name)
        result = environment.step(action)

        if "tool_output" in result.info:
            print("tool:", result.info["tool_output"])
        if "documents" in result.info:
            print("documents:", result.info["documents"])

        if result.done:
            print("answer:", result.answer)
            return

        observation = result.observation


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "experiment":
        config = experiment_config(
            training_episodes=args.episodes,
            seed=args.seed,
        )
        result = run_experiment(config, args.output_dir)
        print(format_metrics(result.metrics))

    elif args.command == "ask":
        ask_question(args.question, args.policy)


if __name__ == "__main__":
    main()
