from adaptive_hitl_agent.experiment import (
    AgentComponents,
    ExperimentConfig,
    load_tasks,
    make_environment_factory,
    run_experiment,
)
from adaptive_hitl_agent.llm import RuleBasedLanguageModel
from adaptive_hitl_agent.policy import TrainingConfig, train_dqn
from adaptive_hitl_agent.retrieval import BM25Retriever
from adaptive_hitl_agent.tools import SafeCalculator


def test_dataset_has_all_routes_in_both_splits() -> None:
    tasks = load_tasks()
    for split in ("train", "test"):
        categories = {task.category for task in tasks if task.split == split}
        assert categories == {"direct", "retrieve", "tool", "human"}


def test_short_dqn_training_returns_a_valid_policy() -> None:
    tasks = [task for task in load_tasks() if task.split == "train"]
    components = AgentComponents(
        retriever=BM25Retriever.from_package_data(),
        language_model=RuleBasedLanguageModel(),
        calculator=SafeCalculator(),
    )
    factory = make_environment_factory(components, ExperimentConfig(training_episodes=40))
    result = train_dqn(tasks, factory, TrainingConfig(episodes=40, seed=3))
    environment = factory(tasks[0])
    action = result.policy.select_action(
        environment.observe(), environment.available_actions()
    )
    assert action in environment.available_actions()
    assert len(result.episode_rewards) == 40


def test_experiment_evaluates_requested_baselines() -> None:
    result = run_experiment(ExperimentConfig(training_episodes=50, seed=11))
    assert {
        "adaptive_rl",
        "fixed_rag",
        "always_retrieve",
        "heuristic_router",
        "no_human",
        "direct_only",
    } == set(result.metrics)
    assert len(result.traces) == 12 * 6
