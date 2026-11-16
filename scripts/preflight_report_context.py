"""Check every full prompt against exact self-hosted tokenisers without generating answers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_coordinate_report import api

from pdbthink.dataset import load_dataset
from pdbthink.util import read_json, stable_hash, write_json


def preflight(dataset: Path, config: Path, output: Path, models: list[str]) -> None:
    _, renders = load_dataset(dataset)
    endpoints = read_json(config)
    for name in models:
        endpoint = endpoints[name]
        if not endpoint.get("tokenize"):
            raise ValueError(f"{name} has no exact tokenisation endpoint")

        def check(render, endpoint=endpoint):
            body = {
                "model": endpoint["model_id"],
                "messages": [
                    {"role": "system", "content": render.system_prompt},
                    {"role": "user", "content": render.user_prompt},
                ],
                "add_generation_prompt": True,
                "chat_template_kwargs": endpoint["extra"].get("chat_template_kwargs", {}),
            }
            response = api(
                {**endpoint, "base_url": endpoint["base_url"].removesuffix("/v1")}, "/tokenize", body
            )
            count = response["count"]
            if not 0 < count < endpoint["context_window"]:
                raise ValueError(f"{render.render_id} does not fit the complete context")
            return {
                "render_id": render.render_id,
                "input_tokens": count,
                "remaining_output_capacity": endpoint["context_window"] - count,
                "tokenize_request_sha256": stable_hash(body),
                "tokenize_response_sha256": stable_hash(response),
            }

        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = list(pool.map(check, renders))
        result = {
            "model": name,
            "context_window": endpoint["context_window"],
            "questions": len(rows),
            "min_input_tokens": min(r["input_tokens"] for r in rows),
            "max_input_tokens": max(r["input_tokens"] for r in rows),
            "rows": rows,
        }
        write_json(output / f"{name}-context-preflight.json", result)
        print(name, result["questions"], result["min_input_tokens"], result["max_input_tokens"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["snowball", "qwen"])
    args = parser.parse_args()
    preflight(args.dataset, args.config, args.output, args.models)
