"""Evalchemy integration.

Evalchemy discovers benchmarks as ``eval/chat_benchmarks/<Name>/eval_instruct.py``
containing a :class:`BaseBenchmark` subclass. Install this benchmark with::

    ln -s $(python -c "import pdbthink.evalchemy_task as m; print(m.__path__[0])") \
          $EVALCHEMY/eval/chat_benchmarks/PDBThink

    python -m eval.eval --model vllm --tasks PDBThink \
        --model_args "pretrained=Qwen/Qwen3-8B" --output_path logs

and point it at a built dataset with ``PDBTHINK_DATASET=<dataset_dir>``.

The lm-eval and Evalchemy imports are deliberately deferred to call time so the
rest of pdbthink -- building, validating, scoring and reporting -- never depends
on Evalchemy being installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

DEFAULT_DATASET_ENV = "PDBTHINK_DATASET"
DEFAULT_MAX_TOKENS = 8192


def _base_benchmark_class():
    from eval.task import BaseBenchmark

    return BaseBenchmark


def _make_benchmark_class():
    BaseBenchmark = _base_benchmark_class()

    class PDBThinkBenchmark(BaseBenchmark):
        """Protein Structural Reasoning Benchmark as an Evalchemy task.

        Single-turn, no tools, deterministic scoring. Every rendered variant of
        every accepted semantic instance becomes one generation request; scoring
        reuses :mod:`pdbthink.scoring` unchanged, so Evalchemy numbers and
        ``structural-reasoning score`` numbers cannot drift apart.
        """

        def __init__(
            self,
            dataset_dir: str | None = None,
            debug: bool = False,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            n_repeat: int = 1,
            families: list[str] | None = None,
            logger: logging.Logger | None = None,
            system_instruction: str | None = None,
        ) -> None:
            super().__init__(logger=logger, system_instruction=system_instruction)
            self.dataset_dir = dataset_dir or os.environ.get(DEFAULT_DATASET_ENV)
            if not self.dataset_dir:
                raise ValueError(
                    "set PDBTHINK_DATASET to a dataset directory built with "
                    "`structural-reasoning build`"
                )
            self.debug = debug
            self.max_new_tokens = max_tokens
            self.n_repeat = n_repeat
            self.families = families

        # -- generation ------------------------------------------------- #
        def generate_responses(self, model) -> dict[str, Any] | None:
            from lm_eval.api.instance import Instance

            from ..dataset import load_dataset

            instances, renders = load_dataset(self.dataset_dir)
            accepted = {
                i.semantic_instance_id for i in instances if i.curation_status != "rejected"
            }
            renders = [r for r in renders if r.semantic_instance_id in accepted]
            if self.families:
                renders = [r for r in renders if r.question_family in set(self.families)]
            renders.sort(key=lambda r: r.render_id)
            if self.debug:
                renders = renders[:2]

            all_outputs = []
            for repeat in range(self.n_repeat):
                requests = []
                for index, render in enumerate(renders):
                    messages = [
                        {"role": "system", "content": render.system_prompt},
                        {"role": "user", "content": render.user_prompt},
                    ]
                    templated = self._prepare_messages(messages, model)
                    request = Instance(
                        "generate_until",
                        {"render_id": render.render_id},
                        (
                            templated,
                            {
                                "do_sample": self.n_repeat > 1,
                                "max_new_tokens": self.max_new_tokens,
                                "temperature": 0.0 if self.n_repeat == 1 else 0.7,
                            },
                        ),
                        index,
                    )
                    request.repeat_idx = repeat
                    request.metadata = {
                        "render_id": render.render_id,
                        "question_family": render.question_family,
                        "answer_schema": render.answer_schema,
                    }
                    requests.append(request)
                self.logger.info(
                    "PDBThink: generating %d responses (repeat %d/%d)",
                    len(requests), repeat + 1, self.n_repeat,
                )
                all_outputs.append(self.compute(model, requests))

            if model.rank != 0:
                return None
            return {
                "dataset_dir": str(self.dataset_dir),
                "render_ids": [r.render_id for r in renders],
                "outputs": [list(group) for group in zip(*all_outputs)],
            }

        # -- scoring ------------------------------------------------------ #
        def evaluate_responses(self, results: dict[str, Any] | None) -> dict[str, Any] | None:
            if results is None:
                return None

            from ..dataset import load_dataset
            from ..evaluation.score import summarise
            from ..scoring import score_response

            instances, renders = load_dataset(results["dataset_dir"])
            by_instance = {i.semantic_instance_id: i for i in instances}
            by_render = {r.render_id: r for r in renders}

            scored: list[dict[str, Any]] = []
            for render_id, completions in zip(results["render_ids"], results["outputs"]):
                render = by_render[render_id]
                instance = by_instance[render.semantic_instance_id]
                parameters = _scoring_parameters(instance)
                for completion_index, text in enumerate(completions):
                    outcome = score_response(
                        text or "", render.answer_schema, render.gold_answer, parameters=parameters
                    )
                    scored.append(
                        {
                            "run_id": "evalchemy",
                            "render_id": render_id,
                            "semantic_instance_id": render.semantic_instance_id,
                            "completion_index": completion_index,
                            "question_family": render.question_family,
                            "protein_group_id": render.protein_group_id,
                            "cluster": (instance.gold_evidence or {}).get("cluster")
                            or render.protein_group_id,
                            "representation": render.representation,
                            "is_rotation_variant": render.is_rotation_variant,
                            "state_order_seed": render.state_order_seed,
                            "answer_schema": render.answer_schema,
                            "score": outcome["score"]["score"],
                            "correct": bool(outcome["score"].get("correct")),
                            "detail": outcome["score"],
                            "parsed_answer": outcome["parsed"],
                            "format_error": outcome["format_error"],
                            "refusal": outcome["refusal"],
                            "truncated": outcome["truncated"],
                            "api_error": False,
                        }
                    )
            summary = summarise(scored)
            summary["scores"] = scored
            return summary

    return PDBThinkBenchmark


def _scoring_parameters(instance) -> dict[str, Any]:
    from ..generators import get_generator

    parameters = dict(instance.question_parameters)
    if instance.answer_schema == "multi_field":
        return parameters
    try:
        generator = get_generator(instance.question_family)
    except KeyError:
        return parameters
    return {**parameters, **generator.prompt_parameters(instance.question_parameters)}


def __getattr__(name: str):
    """Build the benchmark class lazily so importing this module is always safe."""
    if name == "PDBThinkBenchmark":
        return _make_benchmark_class()
    raise AttributeError(name)
