# Adaptive Human-in-the-Loop LLM Agent

[![CI](https://github.com/HadisArefanJazi/adaptive-human-in-the-loop-llm-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/HadisArefanJazi/adaptive-human-in-the-loop-llm-agent/actions/workflows/ci.yml)

A small, reproducible project that trains an LLM agent to decide when it should answer directly, retrieve evidence, use a tool, or ask a human for help.

## How it works

The routing policy is a PyTorch Deep Q-Network (DQN). For each question, it chooses one of four actions:

- `ANSWER_DIRECTLY`
- `RETRIEVE` with local BM25 retrieval
- `USE_TOOL` with a safe calculator
- `ASK_HUMAN`

The reward favors correct answers and applies costs for tokens, retrieval, tool calls, latency, and human intervention. Retrieval and tool use update the agent's state, so the policy can make multi-step decisions such as `RETRIEVE → ANSWER_DIRECTLY`.

The default experiment uses a deterministic language model and a bundled benchmark. It runs on CPU and does not download model weights. A Hugging Face adapter is also included for experiments with a local causal language model.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
python -m adaptive_hitl_agent experiment --episodes 600 --seed 7
```

The experiment saves `metrics.json`, `traces.jsonl`, and `dqn_policy.pt` in `artifacts/latest/`.

## Benchmark results

The bundled benchmark contains 20 training tasks and 12 held-out test tasks. The committed results below use seed 7 and 600 training episodes.

| Policy | Task success | Average reward | Avg. tokens | Retrievals | Tool calls | Human intervention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive RL | **100%** | **0.788** | 14.8 | 0.25 | 0.25 | 25% |
| Fixed RAG gate | 50% | 0.305 | 17.4 | 0.25 | 0.00 | 0% |
| Always retrieve | 50% | 0.193 | 36.2 | 1.00 | 0.00 | 0% |
| Heuristic router | **100%** | **0.788** | 14.8 | 0.25 | 0.25 | 25% |
| Learned policy, no human | 75% | 0.553 | 24.7 | 0.50 | 0.50 | 0% |
| Direct only | 25% | 0.028 | **12.4** | 0.00 | 0.00 | 0% |

These results validate the implementation on a small, controlled dataset. They are not a claim about general LLM performance. Full machine-readable results are available in [`results/benchmark_seed7.json`](results/benchmark_seed7.json).

## Project layout

- `environment.py`: decision process and cost-aware reward
- `policy.py`: DQN, replay buffer, and training loop
- `retrieval.py`: local BM25 retriever
- `tools.py`: AST-based calculator
- `llm.py`: deterministic and Hugging Face model adapters
- `baselines.py`: comparison policies
- `experiment.py`: training, evaluation, and saved artifacts

## Using a Hugging Face model

```python
from adaptive_hitl_agent.experiment import run_experiment
from adaptive_hitl_agent.llm import HuggingFaceLanguageModel

model = HuggingFaceLanguageModel(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    device="cpu",
)
run_experiment(language_model=model, output_dir="artifacts/huggingface")
```

Hugging Face downloads the selected model on first use.

## Limitations

The benchmark is intentionally small. Human responses are simulated with reference answers, latency uses configurable units rather than wall-clock time, and evaluation relies on exact-match scoring. Real deployments need real reviewers, calibrated confidence estimates, representative data, and task-specific evaluation.

## License

MIT
