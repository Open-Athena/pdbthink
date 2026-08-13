# Deploying the review interface

The review server is a dependency-free `ThreadingHTTPServer`. It reads a dataset
directory at startup and appends decisions to one JSONL file, so a deployment
needs exactly three things: the package, the dataset directory, and a writable
path for decisions. There is no database.

What it does *not* have is identity. The interface deliberately serves
curator-only provenance — source entries, release dates, file hashes — and
accepts decisions from any caller, so anything beyond `127.0.0.1` needs a gate.

## Quick tunnel: a URL in two minutes

Good for a review session with colleagues. No Cloudflare account required.

```bash
export PDBTHINK_REVIEW_TOKEN=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
```

```bash
structural-reasoning review --dataset data/datasets/candidates_v1 --decisions data/review_decisions/v1.jsonl --host 127.0.0.1 --port 8787 --no-browser
```

```bash
cloudflared tunnel --url http://127.0.0.1:8787
```

Share `https://<generated>.trycloudflare.com/?token=<token>`. The token is
accepted as a query parameter, a cookie or a bearer header, so the link works
once and the cookie carries the rest of the session.

Bind the server to `127.0.0.1`, not `0.0.0.0`: the tunnel reaches it locally, and
leaving it on all interfaces exposes an unauthenticated path on the LAN that
bypasses nothing but is one misconfiguration away from mattering.

Limits worth knowing: the hostname changes on every restart, `trycloudflare.com`
carries no SLA, and the server still runs on the machine holding the dataset, so
it stops when that machine does.

## Named tunnel with Cloudflare Access: the durable version

Gives a stable hostname, SSO, per-user identity and an audit trail. Requires a
Cloudflare account and a domain on Cloudflare, and the first step opens a
browser, so it cannot be automated end to end.

```bash
cloudflared tunnel login
```

```bash
cloudflared tunnel create pdbthink-review
```

```bash
cloudflared tunnel route dns pdbthink-review review.example.org
```

```bash
cloudflared tunnel run --url http://127.0.0.1:8787 pdbthink-review
```

Then add a Cloudflare Access application for `review.example.org` with an email
or identity-provider policy. Access authenticates at the edge, so the shared
token becomes redundant — though leaving it set costs nothing and keeps the
server safe if the tunnel is ever pointed somewhere else.

With Access in front, populate the curator field from its identity header
(`Cf-Access-Authenticated-User-Email`) instead of the self-reported name, which
turns the decision log into a genuine attribution record.

## Persistence

`decisions.jsonl` is the only mutable state. Keep it on a durable path and commit
it to the repository after a review session: it is the input the dataset builder
consumes with `--decisions ... --accepted-only`, and it is small enough to
version.

## What not to bother with

Rewriting the server as a Cloudflare Worker means moving the dataset into R2 and
the decisions into KV or D1, and reimplementing the API in JavaScript. For a tool
a handful of curators use during a review pass, a tunnel to the machine that
already holds the data is less work and keeps one copy of the truth.
