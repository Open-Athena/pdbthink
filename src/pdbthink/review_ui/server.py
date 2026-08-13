"""Curator review interface (specification section 10).

A dependency-free local HTTP server. It shows, for every proposed instance: an
interactive 3D view with the queried residues, atoms, ligands and coordination
edges highlighted; the exact model-visible prompt; the gold answer; the
underlying continuous measurements and ambiguity margins; the source identifier
and release date (curator-only); the generator version; the representation and
token count; and the acceptance or rejection reasons the generator recorded.

Decisions are appended to a JSONL file that the dataset builder consumes.
"""

from __future__ import annotations

import hmac
import json
import os
import re
import threading
import webbrowser
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ..dataset import load_dataset
from ..schemas import ReviewDecision
from ..util import append_jsonl, read_jsonl, write_json

STATIC = Path(__file__).resolve().parent / "static"
VIEWER_URL = "https://3Dmol.org/build/3Dmol-min.js"
TOKEN_ENV = "PDBTHINK_REVIEW_TOKEN"
TOKEN_COOKIE = "pdbthink_review"

#: Curator-facing shorthand for each family, so the filter reads as more than a
#: code. Deliberately not part of :func:`prompt_fingerprint` — no model sees it.
FAMILY_LABELS = {
    "P01": "chain identifiers",
    "P02": "residue count",
    "P03": "atom coordinates",
    "G01": "distance between atoms",
    "G02": "nearest non-bonded atom",
    "G03": "nearest of candidates",
    "G04": "worst steric clash",
    "S01": "salt-bridge partner",
    "S02": "phosphorylated residue",
    "S03": "buried or exposed",
    "S04": "secondary structure",
    "S05": "chain fold class",
    "S06": "ligand binding site",
    "S07": "metal coordination",
    "S08": "disulfide partner",
    "S09": "chi1 rotamer",
    "I01": "chain interface",
    "N01": "shared contact",
    "T01": "contacts gained/lost",
    "MECH": "mechanistic episode",
}


class ReviewState:
    """Dataset plus decisions, shared by the request handlers."""

    def __init__(self, dataset_dir: str | Path, decisions_path: str | Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.decisions_path = Path(decisions_path)
        self.instances, self.renders = load_dataset(self.dataset_dir)
        self.by_instance = {i.semantic_instance_id: i for i in self.instances}
        self.renders_by_instance: dict[str, list] = {}
        for render in self.renders:
            self.renders_by_instance.setdefault(render.semantic_instance_id, []).append(render)
        self.decisions = self._load_decisions()
        self.lock = threading.Lock()

    def _load_decisions(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.decisions_path):
            out[row["semantic_instance_id"]] = row
        return out

    def record(self, payload: dict[str, Any]) -> dict[str, Any]:
        decision = ReviewDecision(
            semantic_instance_id=payload["semantic_instance_id"],
            decision=payload["decision"],
            reason=payload.get("reason", ""),
            notes=payload.get("notes", ""),
            label_override=payload.get("label_override"),
            curator=payload.get("curator", ""),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        with self.lock:
            self.decisions[decision.semantic_instance_id] = decision.model_dump()
            append_jsonl(self.decisions_path, decision.model_dump())
        return decision.model_dump()

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for instance in self.instances:
            decision = self.decisions.get(instance.semantic_instance_id)
            rows.append(
                {
                    "id": instance.semantic_instance_id,
                    "family": instance.question_family,
                    "family_label": FAMILY_LABELS.get(instance.question_family, ""),
                    "protein": instance.protein_group_id,
                    "source_type": instance.source_type,
                    "entries": instance.source_entries,
                    "status": (decision or {}).get("decision", "pending"),
                    "is_mechanistic": instance.is_mechanistic,
                    "n_renders": len(self.renders_by_instance.get(instance.semantic_instance_id, [])),
                }
            )
        return rows

    def detail(self, instance_id: str) -> dict[str, Any]:
        instance = self.by_instance[instance_id]
        renders = sorted(self.renders_by_instance.get(instance_id, []), key=_render_rank)
        return {
            "instance": json.loads(instance.model_dump_json()),
            "renders": [
                {
                    "render_id": r.render_id,
                    "representation": r.representation,
                    "rotation_seed": r.rotation_seed,
                    "state_order_seed": r.state_order_seed,
                    "is_rotation_variant": r.is_rotation_variant,
                    "input_token_count": r.input_token_count,
                    "tokenizer": r.tokenizer,
                    "atom_count": r.atom_count,
                    "crop": r.crop,
                    "system_prompt": r.system_prompt,
                    "user_prompt": r.user_prompt,
                    "gold_answer": r.gold_answer,
                    "displayed_coordinates_sha256": r.displayed_coordinates_sha256,
                    "structures": _extract_structures(r.user_prompt)
                    if r.representation == "minimal_pdb"
                    else [],
                }
                for r in renders
            ],
            "decision": self.decisions.get(instance_id),
            "highlights": _highlights(instance),
        }


#: Representation order for the curator, most informative first. Alphabetical
#: order would open every mechanistic episode on its context-only control, whose
#: whole point is that it carries no coordinates.
_REPRESENTATION_ORDER = {"minimal_pdb": 0, "normalized_coordinates": 1, "context_only": 2}


def _render_rank(render) -> tuple[int, int, int, int]:
    """Primary variant first, then reordered states, rotations, controls."""
    return (
        _REPRESENTATION_ORDER.get(render.representation, 99),
        int(bool(render.is_rotation_variant)),
        int(render.state_order_seed or 0),
        int(render.rotation_seed or 0),
    )


def _extract_structures(prompt: str) -> list[dict[str, str]]:
    """Pull the PDB blocks out of a rendered prompt for the 3D viewer."""
    out: list[dict[str, str]] = []
    current_label: str | None = None
    lines: list[str] = []
    for line in prompt.splitlines():
        if re.match(r"^Structure( \d+)?:$", line.strip()):
            if current_label and lines:
                out.append({"label": current_label, "pdb": "\n".join(lines)})
            current_label = line.strip().rstrip(":")
            lines = []
            continue
        if current_label is not None:
            if line.startswith(("ATOM", "HETATM", "TER", "END")):
                lines.append(line)
            elif lines and not line.strip():
                out.append({"label": current_label, "pdb": "\n".join(lines)})
                current_label, lines = None, []
    if current_label and lines:
        out.append({"label": current_label, "pdb": "\n".join(lines)})
    return out


RESIDUE_TOKEN = re.compile(r"\b([A-Za-z0-9]):([A-Za-z][A-Za-z0-9]{0,2}?)(-?\d+)\b")


def _highlights(instance) -> list[dict[str, Any]]:
    """Residues to highlight: query parameters, gold answer and evidence."""
    found: dict[str, str] = {}

    def scan(value: Any, role: str) -> None:
        if isinstance(value, str):
            for match in RESIDUE_TOKEN.finditer(value):
                found.setdefault(match.group(0), role)
        elif isinstance(value, dict):
            for v in value.values():
                scan(v, role)
        elif isinstance(value, list):
            for v in value:
                scan(v, role)

    scan(instance.question_parameters, "query")
    scan(instance.gold_answer, "gold")
    scan(instance.gold_evidence, "evidence")
    out = []
    for label, role in found.items():
        chain, _, rest = label.partition(":")
        number = re.search(r"(-?\d+)$", rest)
        if not number:
            continue
        out.append({"label": label, "chain": chain, "resi": int(number.group(1)), "role": role})
    return sorted(out, key=lambda h: (h["role"], h["chain"], h["resi"]))


class Handler(BaseHTTPRequestHandler):
    state: ReviewState
    auth_token: str | None = None

    def log_message(self, *args) -> None:  # noqa: D102 - silence per-request logging
        pass

    # -- authentication ------------------------------------------------- #
    def _authorised(self) -> bool:
        """Check the shared token, if one is configured.

        The interface shows curator-only provenance and accepts decisions, so it
        must not be reachable by anyone who merely knows the URL. A shared token
        is the minimum; put the server behind an identity-aware proxy
        (Cloudflare Access, Tailscale, oauth2-proxy) for real per-curator
        identity and an audit trail.
        """
        if not self.auth_token:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and hmac.compare_digest(header[7:], self.auth_token):
            return True
        query = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        if query and hmac.compare_digest(query, self.auth_token):
            return True
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get(TOKEN_COOKIE)
        return bool(morsel and hmac.compare_digest(morsel.value, self.auth_token))

    def _reject(self) -> None:
        body = b"unauthorised: append ?token=... to the URL\n"
        self.send_response(401)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if not self._authorised():
            self._reject()
            return
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", (STATIC / "review.html").read_bytes())
        elif path == "/app.js":
            self._send(200, "application/javascript", (STATIC / "review.js").read_bytes())
        elif path == "/style.css":
            self._send(200, "text/css", (STATIC / "review.css").read_bytes())
        elif path == "/api/instances":
            self._json(self.state.summary())
        elif path.startswith("/api/instance/"):
            instance_id = path.rsplit("/", 1)[-1]
            if instance_id not in self.state.by_instance:
                self._json({"error": "unknown instance"}, status=404)
            else:
                self._json(self.state.detail(instance_id))
        elif path == "/api/viewer-url":
            self._json({"url": VIEWER_URL})
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if not self._authorised():
            self._reject()
            return
        if urlparse(self.path).path != "/api/decision":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            saved = self.state.record(payload)
        except Exception as exc:  # noqa: BLE001 - reported to the curator
            self._json({"error": str(exc)}, status=400)
            return
        self._json({"ok": True, "decision": saved})

    def _json(self, payload: Any, status: int = 200) -> None:
        self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Remember a token supplied in the query string so the rest of the
        # session works from an ordinary link.
        if self.auth_token and parse_qs(urlparse(self.path).query).get("token"):
            self.send_header(
                "Set-Cookie",
                f"{TOKEN_COOKIE}={self.auth_token}; Path=/; HttpOnly; SameSite=Lax",
            )
        self.end_headers()
        self.wfile.write(body)


def serve(
    dataset_dir: str | Path,
    decisions_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    open_browser: bool = True,
    auth_token: str | None = None,
) -> None:
    state = ReviewState(dataset_dir, decisions_path)
    auth_token = auth_token or os.environ.get(TOKEN_ENV) or None
    handler = type("BoundHandler", (Handler,), {"state": state, "auth_token": auth_token})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    if auth_token:
        url = f"{url}?token={auth_token}"
    print(f"curator review interface: {url}")
    print(f"  dataset   : {state.dataset_dir}")
    print(f"  decisions : {state.decisions_path}")
    print(f"  instances : {len(state.instances)} ({len(state.decisions)} already decided)")
    if auth_token:
        print("  auth      : shared token required")
    elif host not in ("127.0.0.1", "localhost"):
        print(
            "  auth      : NONE, and bound beyond localhost. This interface serves "
            "curator-only provenance and accepts decisions from anyone who can reach "
            f"it; pass --auth-token or set {TOKEN_ENV}."
        )
    print("  press Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping review server")
    finally:
        server.server_close()


def export_decisions(decisions_path: str | Path, output_path: str | Path) -> int:
    """Export the decision log as a JSON array (section 10)."""
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(decisions_path):
        latest[row["semantic_instance_id"]] = row
    rows = [latest[k] for k in sorted(latest)]
    write_json(output_path, rows)
    return len(rows)
