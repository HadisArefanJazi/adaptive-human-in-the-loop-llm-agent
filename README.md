# Adaptive Human-in-the-Loop LLM Agent

[![CI](https://github.com/HadisArefanJazi/adaptive-human-in-the-loop-llm-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/HadisArefanJazi/adaptive-human-in-the-loop-llm-agent/actions/workflows/ci.yml)

A PyTorch DQN router that learns when an assistant should answer, retrieve evidence, use a calculator, or ask a human. The objective is to solve the task without spending more resources than necessary.

The default experiment runs offline on CPU. A deterministic model supplies answers, a local BM25 index supplies documents, and an idealized reviewer supplies human responses. This keeps routing decisions and their costs inspectable without downloading model weights. An optional Hugging Face adapter supports experiments with a causal language model.

## Run the project

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest --cov=adaptive_hitl_agent --cov-report=term-missing --cov-fail-under=90
python -m adaptive_hitl_agent experiment --episodes 600 --seed 7
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

The experiment writes `metrics.json`, `traces.jsonl`, and `dqn_policy.pt` to `artifacts/latest/`. Metrics include the resolved training configuration, software versions, and dataset hashes. Traces record each task's actions, answer, correctness, and resource use.

## Routing and reward

Each question starts a short episode. `RETRIEVE` adds BM25 documents; `USE_TOOL` adds a calculator result. `ANSWER_DIRECTLY` generates an answer using whatever evidence is available, and `ASK_HUMAN` requests a reviewer answer. The two answer actions end the episode.

The state has seven features: answer confidence with the current context, arithmetic and ambiguity signals, normalized question length, the presence of retrieved documents and tool output, and episode progress. Confidence and routing signals are heuristics, not calibrated probabilities. Constructing a state does not query BM25. Retrieval runs only when selected, and each retrieval or tool action can run once. The final step is reserved for an answer or review.

A correct final answer earns `1.0`; an incorrect one earns `-0.25`. Every step also pays for its resource use:

```text
cost = 0.002 * estimated tokens
     + 0.08  * retrieval calls
     + 0.04  * tool calls
     + 0.45  * human calls
     + 0.02  * latency units
```

Tokens are word-based estimates, and latency uses fixed units rather than wall-clock measurements. They make the cost trade-off reproducible; they are not API billing or production timing data. The step limit bounds episode length, not total monetary cost.

## DQN training

The Q-network is a `7 -> 32 -> 32 -> 4` multilayer perceptron with ReLU hidden layers. It uses a replay buffer of 4,000 transitions, batches of 32, Adam with learning rate `0.002`, Huber loss, and gradient clipping at norm `5`.

Exploration decreases linearly from epsilon `0.9` to `0.05` across training episodes. Both action selection and Bellman targets mask unavailable actions. Terminal transitions use only their immediate reward; other transitions bootstrap from the target network with discount `0.95`. The target network is copied from the online network every 40 completed episodes. Reported episode rewards are undiscounted sums, separate from the discounted training objective.

The checkpoint stores feature and action names alongside the weights. Load it with:

```python
from adaptive_hitl_agent.dqn import DQNPolicy

policy = DQNPolicy.load("artifacts/latest/dqn_policy.pt")
```

Checkpoints must match the current state and action schema. Training starts from scratch; the saved checkpoint is for inference, not optimizer-state resumption.

## Evaluation

The bundled data has 20 training questions and 12 held-out test questions, balanced across direct answers, retrieval, arithmetic, and human review. The following run uses 600 training episodes and seed 7:

| Policy | System success | Autonomous success | Answer precision | Answer coverage | Avg. reward | Retrievals | Tool calls | Human review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive RL | 100% | 75% | 100% | 75% | 0.783 | 0.25 | 0.33 | 25% |
| Confidence-gated RAG | 50% | 50% | 100% | 50% | 0.230 | 0.75 | 0.00 | 0% |
| Always retrieve | 50% | 50% | 100% | 50% | 0.193 | 1.00 | 0.00 | 0% |
| Heuristic router | 100% | 75% | 100% | 75% | 0.788 | 0.25 | 0.25 | 25% |
| Learned policy, no human | 75% | 75% | 100% | 75% | 0.547 | 0.50 | 0.58 | 0% |
| Direct only | 25% | 25% | 100% | 25% | 0.028 | 0.00 | 0.00 | 0% |

System success counts correct final answers, including reviewer answers. Autonomous success counts correct answers without review, divided by all tasks. Answer precision is accuracy among substantive autonomous answers; coverage is the fraction of all tasks receiving such an answer. Empty answers, `I don't know`, `TOOL_ERROR`, and `None` are treated as abstentions after normalization. Precision is reported as zero when there are no substantive autonomous answers, so it should always be read alongside coverage. Retrievals and tool calls are averages per task.

The confidence-gated baseline retrieves below confidence `0.5`, then answers. The heuristic prioritizes ambiguity for human review, arithmetic for tool use, and low confidence for retrieval. The no-human comparison masks human review on the already-trained DQN; it is not a separately retrained policy.

Across seeds `1, 3, 5, 7, 11, 13, 17, 19`, the DQN achieved 100% system success and 75% autonomous success in every run. Its mean reward was `0.78723`, with sample standard deviation `0.00171`, compared with `0.78783` for the heuristic. Seed 7 made one unnecessary tool call. These results show that the DQN learns the routing pattern; they do not establish an advantage over the heuristic. Repeated seeds measure training variation on this fixed test set, not uncertainty on unseen populations.

Reproduce both result files with:

```bash
python scripts/benchmark.py
```

This writes per-seed traces and checkpoints under `artifacts/benchmark/`, plus [`results/benchmark_seed7.json`](results/benchmark_seed7.json) and [`results/benchmark_multiseed.json`](results/benchmark_multiseed.json). The additional full-metrics export defaults to seed 7 when included; use `--reference-seed N` to select another included seed. This controls the export only, not which runs enter the aggregate. No best-seed selection is performed. Floating-point training details can vary with PyTorch versions and hardware.

## Code layout

| Module | Responsibility |
| --- | --- |
| `environment.py` | Episode state, legal actions, transitions, and costs |
| `dqn.py` | Q-network, replay, exploration, target updates, and checkpoints |
| `retrieval.py` | BM25 indexing and ranking |
| `tools.py` | Arithmetic extraction and bounded AST evaluation |
| `llm.py` | Deterministic model and optional Hugging Face adapter |
| `human.py` | Reviewer interface and oracle simulator |
| `baselines.py` | Fixed policies and the no-human ablation |
| `evaluation.py` | Success, precision, coverage, category metrics, and traces |
| `experiment.py` | Dataset loading, component assembly, and artifact writing |

Configuration lives in `ExperimentConfig`, `TrainingConfig`, and `RewardConfig`. The calculator permits only numeric constants and supported arithmetic operators, with limits on input length, expression complexity, exponents, and intermediate magnitudes. Inputs must be a complete arithmetic expression, optionally prefixed by `Calculate`, `Compute`, or `What is`. Unsupported notation and mixed prose are rejected rather than partially evaluated. A failed calculation is recorded but does not replace otherwise valid model evidence.

## Optional Hugging Face model

```bash
python -m pip install -e ".[hf]"
```

```python
from adaptive_hitl_agent.experiment import run_experiment
from adaptive_hitl_agent.llm import HuggingFaceLanguageModel

model = HuggingFaceLanguageModel(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",
)
run_experiment(language_model=model, output_dir="artifacts/huggingface")
```

The selected model is downloaded on first use. This adapter uses a heuristic confidence prior and estimated costs; the benchmark table above is not an evaluation of that model. Adapter tests use local fakes rather than downloaded weights.

## Scope and limitations

The deterministic model uses a small answer dictionary and the answer field of the top relevant document. It does not learn language generation or extract answers from prose. Test questions are held out from router training, but their answer sources remain available to the simulated model and retriever.

`OracleHumanReviewer` returns the reference answer. A different reviewer can be passed to `run_experiment(human_reviewer=...)`, and its answer is scored rather than silently replaced by the reference. A real backend must not use benchmark labels and would need measured latency, costs, availability, and reviewer quality.

The dataset is small and synthetic, and exact-match scoring is intended for short answers. Broader evaluation would need paraphrases, overlapping routing cues, irrelevant evidence, reviewer errors, and representative questions. This project is a controlled routing experiment, not a production human-review service.

## License

MIT
