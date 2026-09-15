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
python -m adaptive_hitl_agent experiment --episodes 600 --seed 7
```

seed 7 results:

| policy | system success | autonomous success | precision | coverage | avg. reward | retrievals | tool calls | human |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adaptive_rl | 100.0% | 70.0% | 100.0% | 70.0% | 0.746 | 0.32 | 0.32 | 30.0% |
| confidence_rag | 38.0% | 38.0% | 73.1% | 52.0% | 0.051 | 0.94 | 0.00 | 0.0% |
| always_retrieve | 38.0% | 38.0% | 73.1% | 52.0% | 0.042 | 1.00 | 0.00 | 0.0% |
| heuristic_router | 100.0% | 70.0% | 100.0% | 70.0% | 0.746 | 0.32 | 0.32 | 30.0% |
| no_human | 70.0% | 70.0% | 83.3% | 84.0% | 0.464 | 0.62 | 0.62 | 0.0% |
| direct_only | 6.0% | 6.0% | 100.0% | 6.0% | -0.212 | 0.00 | 0.00 | 0.0% |

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
python -m adaptive_hitl_agent experiment --episodes 600 --seed 7
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

training and test questions are separated.

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

## Limitations

the benchmark is intentionally small and synthetic. the default language model and human reviewer are deterministic simulations, and latency/token costs are estimates rather than real production measurements.

## License

MIT
