# Adaptive human-in-the-loop llm agent

a small reinforcement-learning project that learns when an ai assistant should:

* answer directly,
* retrieve external evidence,
* use a calculator,
* or escalate to a human.

the goal is not only to answer correctly, but to do so while minimizing resource cost.

## Core idea

each question is treated as a short routing episode:

```text
question
   ↓
7-feature state
   ↓
dqn router
   ↓
answer | retrieve | tool | human
   ↓
correctness reward − resource cost
```

the router learns which action is appropriate from interaction with the environment.

## State

the policy observes seven features:

```text
direct model confidence
arithmetic signal
ambiguity signal
question length
retrieval available
tool result available
episode progress
```

these features form the state given to the dqn.

## Actions

the agent has four possible actions:

```text
answer_directly
retrieve
use_tool
ask_human
```

`retrieve` and `use_tool` update the state and allow another decision.

`answer_directly` and `ask_human` terminate the episode.

unavailable actions are masked during both inference and dqn training.

## Reward

a correct final answer receives:

```text
+1.0
```

an incorrect answer receives:

```text
-0.25
```

the agent also pays for resource usage:

```text
cost =
    0.002 * estimated_tokens
  + 0.08  * retrieval_calls
  + 0.04  * tool_calls
  + 0.45  * human_calls
  + 0.02  * latency_units
```

therefore:

```text
reward = answer_quality - resource_cost
```

this encourages the policy to use expensive resources only when necessary.

## DQN

the routing policy is a small pytorch network:

```text
7 → 32 → 32 → 4
```

training uses:

* experience replay,
* epsilon-greedy exploration,
* a target network,
* huber loss,
* gradient clipping,
* masked bellman targets.

default hyperparameters:

```text
replay capacity:        4000
batch size:             32
discount factor:        0.95
learning rate:          0.002
epsilon:                0.90 → 0.05
target update interval: 40 episodes
```

## Components

the experiment intentionally uses lightweight components so the routing behavior remains easy to inspect.

```text
rule_based_language_model
    deterministic model for reproducible experiments

bm25_retriever
    local document retrieval

safe_calculator
    bounded arithmetic evaluation using python ast

oracle_human_reviewer
    simulated human reviewer for benchmark experiments
```

an optional hugging face language-model adapter is also included.

## Results

default experiment:

```bash
python -m adaptive_hitl_agent experiment --episodes 3000 --seed 7
```

seed 7 results:

| policy | system success | autonomous success | precision | coverage | avg. reward | retrievals | tool calls | human |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive_rl | 99.5% | 68.0% | 99.3% | 68.5% | 0.728 | 0.34 | 0.34 | 31.5% |
| confidence_rag | 34.5% | 34.5% | 56.1% | 61.5% | -0.001 | 0.98 | 0.00 | 0.0% |
| always_retrieve | 34.5% | 34.5% | 56.1% | 61.5% | -0.004 | 1.00 | 0.00 | 0.0% |
| heuristic_router | 99.5% | 68.0% | 99.3% | 68.5% | 0.728 | 0.34 | 0.34 | 31.5% |
| no_human | 68.0% | 68.0% | 75.1% | 90.5% | 0.431 | 0.65 | 0.65 | 0.0% |
| direct_only | 1.5% | 1.5% | 100.0% | 1.5% | -0.273 | 0.00 | 0.00 | 0.0% |

the learned policy successfully discovers the intended routing behavior.

the heuristic performs similarly on this small structured dataset, and obviously these results demonstrate successful policy learning rather than an advantage of rl over heuristics.

## Installation

python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -e ".[dev]"
```

run linting and tests:

```bash
python -m ruff check .

python -m pytest \
  --cov=adaptive_hitl_agent \
  --cov-report=term-missing \
  --cov-fail-under=90
```

run the experiment:

```bash
python -m adaptive_hitl_agent experiment --episodes 3000 --seed 7
```

artifacts are written to:

```text
artifacts/latest/
├── metrics.json
├── traces.jsonl
└── dqn_policy.pt
```

## Project structure

```text
adaptive_hitl_agent/
├── types.py          # shared data structures
├── records.py        # record comparison, display, and mutation rules
├── environment.py    # state, actions, rewards, episode logic
├── dqn.py            # dqn policy and training
├── baselines.py      # comparison routing policies
├── retrieval.py      # bm25 retrieval
├── tools.py          # safe calculator
├── llm.py            # language-model interfaces
├── human.py          # human-review interface
├── evaluation.py     # evaluation metrics
├── experiment.py     # experiment orchestration
├── cli.py            # command-line interface
└── data/             # benchmark tasks and documents

tests/
scripts/
results/
```

## Benchmark

the bundled benchmark contains four task categories:

```text
direct
retrieval
tool
human
```

The benchmark contains 500 training questions and 200 test questions. The agent trains for 3,000 episodes, and each policy is evaluated on the test set.

for multi-seed benchmarking:

```bash
python scripts/benchmark.py
```

reference results are stored in:

```text
results/
```

## Optional hugging face model

install:

```bash
python -m pip install -e ".[hf]"
```

example:

```python
from adaptive_hitl_agent.experiment import run_experiment
from adaptive_hitl_agent.llm import hugging_face_language_model

model = hugging_face_language_model(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",
)

run_experiment(
    language_model=model,
    output_dir="artifacts/huggingface",
)
```

## Limitations and Future Work

This project is a proof of concept, and several parts of the environment are heuristic or manually defined.

In a future version, I would replace these simplified parts with more realistic data and models, and use **LangGraph** to organize the multi-step workflow.

## License

MIT
