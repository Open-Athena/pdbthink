"""Evalchemy task adapter.

The package directory doubles as the Evalchemy benchmark directory: symlink it
into ``eval/chat_benchmarks/PDBThink`` and Evalchemy will import
``eval_instruct.PDBThinkBenchmark`` from here.
"""

from .eval_instruct import DEFAULT_DATASET_ENV

__all__ = ["DEFAULT_DATASET_ENV"]
