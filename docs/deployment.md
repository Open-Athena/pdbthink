# Sharing the review interface

The review server is a dependency-free `ThreadingHTTPServer`. It reads a dataset
directory at startup and appends decisions to one JSONL file, so it needs only
the package, the dataset directory, and a writable path for decisions. There is
no database and no build step.

What it does not have is identity. The interface deliberately serves
curator-only provenance — source entries, release dates, file hashes — and
accepts decisions from any caller. Anything reachable beyond `127.0.0.1` needs a
real gate in front of it.

## Recommended: SSH port forwarding

For a handful of curators this is the right answer. Each person authenticates
with credentials they already have, nothing new is exposed, and access is
revoked by removing their key.

Run the server bound to localhost on the machine that holds the dataset:

```bash
structural-reasoning review --dataset data/datasets/candidates_v1 --decisions data/review_decisions/v1.jsonl --host 127.0.0.1 --port 8787 --no-browser
```

Each curator forwards the port from their own machine:

```bash
ssh -L 8787:localhost:8787 user@review-host
```

They then open <http://localhost:8787>. Decisions land in the one JSONL file on
the host, so there is a single copy of the truth and no synchronisation to think
about.

## If you need a hosted URL

Put an identity-aware proxy in front of it — Cloudflare Access, Tailscale, or
`oauth2-proxy` — so authentication happens at the edge against your existing
identity provider, with per-person identity and an audit trail. This requires an
account and a domain you control, and the setup involves an interactive browser
login, so it is not something that can be scripted end to end.

With such a proxy in front, populate the curator field from its identity header
(Cloudflare Access sends `Cf-Access-Authenticated-User-Email`) instead of the
self-reported name. That turns the decision log into a genuine attribution
record, which is the point of having one.

## The shared token is a floor, not a gate

`--auth-token` / `PDBTHINK_REVIEW_TOKEN` exists so that a server which ends up
bound beyond localhost is not wide open, and the startup banner warns when one
is missing. It is deliberately weak: one secret for everyone, no expiry, no
revocation, no attribution, and it travels in the URL if you pass it as a query
parameter — which means browser history, referrer headers and screenshots.

Treat it as a seatbelt against misconfiguration, never as the thing that makes
an endpoint safe to publish.

## The review node

A long-running instance is set up on a private host as systemd user services:

| unit | what it does |
| --- | --- |
| `pdbthink-review.service` | serves the interface on port 8787, restarts on failure |
| `pdbthink-sync.timer` | every five minutes, commits and pushes `data/review_decisions/v1.jsonl` if it changed |

Both run under `loginctl enable-linger`, so they survive logout and reboot. The
token lives in `~/.config/pdbthink/review.env` (mode 600), outside the
repository. The service sets `CUDA_VISIBLE_DEVICES=` explicitly: it is CPU-only
and shares the host with GPU work.

The sync script stages only the decisions file, rebases onto `origin/main`, and
aborts rather than forcing if the rebase conflicts — so a curation session can
never rewrite anything else in the repository.

Rebuilding the dataset on that host reproduced `instances.jsonl` byte for byte
from freshly downloaded structures, which is the determinism requirement holding
across machines rather than just across runs on one.

## Persistence

`decisions.jsonl` is the only mutable state. Keep it on a durable path and commit
it after a review session: it is the input the builder consumes with
`--decisions ... --accepted-only`, and it is small enough to version.

## What not to bother with

Rewriting the server as a serverless function means moving the dataset into
object storage and the decisions into a key-value store, and reimplementing the
API. For a tool a few curators use during a review pass, forwarding a port to
the machine that already holds the data is less work and keeps one copy of the
truth.
