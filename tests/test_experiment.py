from adaptive_hitl_agent.experiment import (
    agent_components,
    experiment_config,
    load_tasks,
    make_environment_factory,
    run_experiment,
)
from adaptive_hitl_agent.human import oracle_human_reviewer
from adaptive_hitl_agent.llm import rule_based_language_model
from adaptive_hitl_agent.dqn import training_config, train_dqn
from adaptive_hitl_agent.retrieval import bm25_retriever
from adaptive_hitl_agent.tools import safe_calculator


def components() -> agent_components:
    return agent_components(
        retriever=bm25_retriever.from_package_data(),
        language_model=rule_based_language_model(),
        calculator=safe_calculator(),
        human_reviewer=oracle_human_reviewer(),
    )


def test_dataset_has_all_routes_in_both_splits() -> None:
    tasks = load_tasks()
    for split in ("train", "test"):
        categories = {task.category for task in tasks if task.split == split}
        assert categories == {"direct", "retrieve", "tool", "human"}


def test_short_dqn_training_returns_a_valid_policy() -> None:
    tasks = [task for task in load_tasks() if task.split == "train"]
    factory = make_environment_factory(
        components(),
        experiment_config(training_episodes=40),
    )
    result = train_dqn(tasks, factory, training_config(episodes=40, seed=3))
    environment = factory(tasks[0])
    action = result.policy.select_action(
        environment.observe(),
        environment.available_actions(),
    )

    assert action in environment.available_actions()
    assert len(result.episode_rewards) == 40


def test_experiment_evaluates_all_comparison_policies() -> None:
    result = run_experiment(experiment_config(training_episodes=50, seed=11))
    assert {
        "adaptive_rl",
        "confidence_rag",
        "always_retrieve",
        "heuristic_router",
        "no_human",
        "direct_only",
    } == set(result.metrics)
    assert len(result.traces) == 12 * 6


def test_seed7_benchmark_is_reproducible() -> None:
    result = run_experiment(experiment_config(training_episodes=600, seed=7))
    adaptive = result.metrics["adaptive_rl"]

    assert adaptive.system_success == 1.0
    assert adaptive.autonomous_success == 0.75
    assert adaptive.intervention_frequency == 0.25


def test_bundled_dataset_is_balanced_and_held_out() -> None:
    tasks = load_tasks()
    assert len({task.task_id for task in tasks}) == len(tasks)
    assert len({task.question.casefold().strip() for task in tasks}) == len(tasks)
    for split, expected in (("train", 5), ("test", 3)):
        counts: dict[str, int] = {}
        for task in tasks:
            if task.split == split:
                counts[task.category] = counts.get(task.category, 0) + 1
        assert counts == {
            route: expected for route in ("direct", "retrieve", "tool", "human")
        }


def test_experiment_records_runtime_and_data_provenance() -> None:
    result = run_experiment(experiment_config(training_episodes=1))
    assert result.metadata["package_version"] == "0.2.0"
    assert result.metadata["platform"]
    assert result.metadata["training_device"] == "cpu"
    assert all(len(value) == 64 for value in result.metadata["data_sha256"].values())
