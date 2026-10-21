"""adaptive human-in-the-loop agent with resource-aware rl routing."""

import sys as _sys

from . import core, evaluate, main
from .dqn import dqn_policy, training_config, train_dqn
from .core import assistance_environment, reward_config_type
from .core import action_type, human_reviewer, oracle_human_reviewer, task_record

__all__ = [
    "action_type",
    "assistance_environment",
    "dqn_policy",
    "human_reviewer",
    "oracle_human_reviewer",
    "reward_config_type",
    "task_record",
    "training_config",
    "train_dqn",
]

__version__ = "0.2.0"

# Keep documented imports and old pickled records working after consolidation.
for _name, _module in {
    "environment": core,
    "experiment": evaluate,
    "cli": main,
    "types": core,
    "retrieval": core,
    "tools": core,
    "llm": core,
    "human": core,
    "baselines": evaluate,
    "evaluation": evaluate,
}.items():
    _sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
del _name, _module, _sys
