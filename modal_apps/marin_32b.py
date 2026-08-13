"""Serve Marin 32B behind an OpenAI-compatible endpoint on Modal.

Marin 32B is not hosted by any inference provider the benchmark already speaks
to, so it gets served here and evaluated through the ordinary ``openai_chat``
path. Two properties of this model shape the run and are worth stating before
anyone reads a score:

* **It is a base model.** ``marin-community/marin-32b-instruct`` does not
  exist. A base model has never been trained to obey "put the answer after
  FINAL:", so a low score may be format non-compliance rather than an inability
  to read coordinates. The scorer counts format errors separately; read that
  number first.

* **Its context is short.** ``config.json`` declares
  ``max_position_embeddings: 4096`` while its llama3 rope_scaling block claims
  an original length of 8192 with factor 8. Those disagree. We serve at
  ``MAX_MODEL_LEN`` and let ``--max-input-tokens`` drop the prompts that do not
  fit, so coverage is explicit rather than silently truncated.

Deploy, then point a model config at the printed URL:

    modal deploy modal_apps/marin_32b.py
"""

import modal

MODEL_NAME = "marin-community/marin-32b-base"
MODEL_REVISION = "main"
#: Serve well beyond the declared 4096 to exercise the rope scaling. Prompts
#: above this are skipped by the runner rather than truncated.
MAX_MODEL_LEN = 32768
N_GPU = 2
GPU_TYPE = f"H100:{N_GPU}"
MINUTES = 60

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]==0.34.4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)

hf_cache = modal.Volume.from_name("pdbthink-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("pdbthink-vllm-cache", create_if_missing=True)

app = modal.App("pdbthink-marin-32b")


@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    scaledown_window=15 * MINUTES,
    timeout=60 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=30 * MINUTES)
def serve() -> None:
    import subprocess

    # A base model ships no chat template, so the OpenAI chat endpoint has
    # nothing to render messages with. Concatenating the turns reproduces the
    # plain-text prompt the benchmark would send to a completion endpoint.
    with open("/root/chat_template.jinja", "w") as handle:
        handle.write("{% for message in messages %}{{ message['content'] }}\n\n{% endfor %}")

    subprocess.Popen(
        [
            "vllm", "serve", MODEL_NAME,
            "--revision", MODEL_REVISION,
            "--served-model-name", MODEL_NAME,
            "--host", "0.0.0.0", "--port", "8000",
            "--tensor-parallel-size", str(N_GPU),
            "--max-model-len", str(MAX_MODEL_LEN),
            # A base model has no chat template of its own; supply a plain one so
            # the OpenAI chat endpoint works at all.
            "--chat-template", "/root/chat_template.jinja",
            "--gpu-memory-utilization", "0.92",
        ]
    )
