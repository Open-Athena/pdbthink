"""Symmetric encryption of curated answers (see ``docs/contamination.md``).

Used for the mechanistic episodes only. Their claims -- the paper-derived
residues and, crucially, the multiple-choice mechanism letter -- cannot be
recomputed from coordinates, so a leak of those is permanent in a way that a
leak of the automatic families' gold answers is not.

**The passphrase is committed next to the ciphertext, so this is obfuscation,
not security.** Anyone who wants the answers has them. What it buys is that the
plaintext no longer appears in a crawlable file, so the benchmark cannot be
memorised by a model that ingested the repository. That is the actual threat.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

#: Committed on purpose. See the module docstring before "fixing" this.
PASSPHRASE = "pdbthink-curated-answers-not-a-secret"
CIPHER = "AES256"


class EncryptionUnavailable(RuntimeError):
    """gpg is not installed, so curated answers cannot be read."""


def _gpg() -> str:
    path = shutil.which("gpg") or shutil.which("gpg2")
    if not path:
        raise EncryptionUnavailable(
            "gpg is required to read the curated mechanistic answers. Install it "
            "(apt install gnupg / brew install gnupg), or run with only the "
            "automatic question families."
        )
    return path


def encrypt_json(payload: Any, path: str | Path, *, passphrase: str = PASSPHRASE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plaintext = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    result = subprocess.run(
        [
            _gpg(), "--batch", "--yes", "--quiet",
            "--passphrase", passphrase, "--pinentry-mode", "loopback",
            "--symmetric", "--cipher-algo", CIPHER,
            "--output", str(path),
        ],
        input=plaintext,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gpg encryption failed: {result.stderr.decode('utf-8', 'replace')}")
    return path


def decrypt_json(path: str | Path, *, passphrase: str = PASSPHRASE) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"encrypted payload not found: {path}")
    result = subprocess.run(
        [
            _gpg(), "--batch", "--quiet",
            "--passphrase", passphrase, "--pinentry-mode", "loopback",
            "--decrypt", str(path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gpg decryption failed: {result.stderr.decode('utf-8', 'replace')}")
    return json.loads(result.stdout.decode("utf-8"))
