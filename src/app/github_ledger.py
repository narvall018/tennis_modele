"""Durable bet ledger stored in the GitHub repository itself.

Streamlit Community Cloud gives an app an ephemeral filesystem: the SQLite
ledger is wiped on every redeploy, including the one the weekly data workflow
triggers. Bets recorded there would vanish without warning, which is the one
thing a ledger must never do.

So the durable copy lives in the repository, written through the GitHub Contents
API with a fine-grained token that needs exactly one permission: *Contents —
Read and write*.

Two decisions make this safe rather than merely possible:

* **A dedicated branch.** Committing to the deployed branch would restart the app
  on every recorded bet, since Cloud redeploys on push. The ledger branch is
  never deployed, so writing to it is invisible to the running app.
* **Optimistic concurrency.** Every write sends the blob SHA it read. If anything
  changed meanwhile GitHub rejects it, and the caller retries against the new
  content instead of overwriting someone else's bet.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_ROOT = "https://api.github.com"
LEDGER_PATH = "ledger/bets.json"
LEDGER_BRANCH = "ledger"
WRITE_ATTEMPTS = 3


@dataclass(frozen=True)
class GitHubConfig:
    token: str
    repository: str
    branch: str = LEDGER_BRANCH
    path: str = LEDGER_PATH

    @property
    def configured(self) -> bool:
        return bool(self.token and "/" in self.repository)


def load_config(root: Path) -> tuple[GitHubConfig, str]:
    """Read the token and repository from secrets or environment."""
    import os

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repository = os.environ.get("GITHUB_REPO", "").strip()
    source = "variables d'environnement"
    if not token:
        try:
            import streamlit as st

            token = str(st.secrets.get("GITHUB_TOKEN", "")).strip()
            repository = str(st.secrets.get("GITHUB_REPO", repository)).strip()
            source = "secrets Streamlit"
        except Exception:  # noqa: BLE001 - secrets are optional
            pass
    if not repository:
        # Fall back to the checkout's own origin, so a local run needs no config.
        head = root / ".git" / "config"
        if head.exists():
            text = head.read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if "github.com" in line and "url" in line:
                    repository = line.split("github.com")[-1].strip(" /:").removesuffix(".git")
                    break
    return GitHubConfig(token=token, repository=repository), source


def _headers(config: GitHubConfig) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def ensure_branch(config: GitHubConfig) -> tuple[bool, str]:
    """Create the ledger branch off the default branch if it does not exist."""
    base = f"{API_ROOT}/repos/{config.repository}"
    try:
        existing = requests.get(
            f"{base}/git/ref/heads/{config.branch}", headers=_headers(config), timeout=30
        )
        if existing.status_code == 200:
            return True, "branche présente"
        repository = requests.get(base, headers=_headers(config), timeout=30)
        if not repository.ok:
            return False, f"dépôt inaccessible ({repository.status_code})"
        default = repository.json().get("default_branch", "main")
        head = requests.get(
            f"{base}/git/ref/heads/{default}", headers=_headers(config), timeout=30
        )
        if not head.ok:
            return False, f"branche par défaut illisible ({head.status_code})"
        created = requests.post(
            f"{base}/git/refs", headers=_headers(config), timeout=30,
            json={"ref": f"refs/heads/{config.branch}",
                  "sha": head.json()["object"]["sha"]},
        )
        if created.status_code in (200, 201):
            return True, f"branche '{config.branch}' créée"
        return False, f"création refusée ({created.status_code}): {created.text[:160]}"
    except requests.RequestException as error:
        return False, f"réseau: {error}"


def read_ledger(config: GitHubConfig) -> tuple[list[dict[str, Any]], str | None, str]:
    """Return the recorded bets, the blob SHA, and a status message."""
    url = f"{API_ROOT}/repos/{config.repository}/contents/{config.path}"
    try:
        response = requests.get(
            url, headers=_headers(config), params={"ref": config.branch}, timeout=30
        )
    except requests.RequestException as error:
        return [], None, f"réseau: {error}"
    if response.status_code == 404:
        return [], None, "carnet vide (fichier pas encore créé)"
    if not response.ok:
        return [], None, f"lecture refusée ({response.status_code})"
    payload = response.json()
    try:
        decoded = base64.b64decode(payload["content"]).decode("utf-8")
        bets = json.loads(decoded) if decoded.strip() else []
    except (KeyError, ValueError, UnicodeDecodeError) as error:
        return [], payload.get("sha"), f"contenu illisible: {error}"
    if not isinstance(bets, list):
        return [], payload.get("sha"), "contenu inattendu: une liste était attendue"
    return bets, payload.get("sha"), f"{len(bets)} pari(s) lus"


def _write(config: GitHubConfig, bets: list[dict[str, Any]], sha: str | None,
           message: str) -> tuple[bool, str]:
    url = f"{API_ROOT}/repos/{config.repository}/contents/{config.path}"
    body: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(
            (json.dumps(bets, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        ).decode("ascii"),
        "branch": config.branch,
    }
    if sha:
        body["sha"] = sha
    try:
        response = requests.put(url, headers=_headers(config), json=body, timeout=30)
    except requests.RequestException as error:
        return False, f"réseau: {error}"
    if response.status_code in (200, 201):
        return True, "écrit"
    if response.status_code == 409:
        return False, "conflit"
    if response.status_code in (401, 403):
        return False, (
            "écriture refusée: le token doit avoir la permission "
            "« Contents: Read and write » sur ce dépôt"
        )
    return False, f"écriture refusée ({response.status_code}): {response.text[:160]}"


def append_bet(config: GitHubConfig, bet: dict[str, Any]) -> tuple[bool, str]:
    """Append one bet, retrying if the file moved under us."""
    if not config.configured:
        return False, "GitHub non configuré: GITHUB_TOKEN et GITHUB_REPO manquants"
    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        **{key: (None if pd.isna(value) else value) if not isinstance(value, str) else value
           for key, value in bet.items()},
    }
    for attempt in range(WRITE_ATTEMPTS):
        bets, sha, _ = read_ledger(config)
        bets.append(record)
        ok, message = _write(
            config, bets, sha,
            f"chore(ledger): {record.get('sport', '?')} · {record.get('pari', '?')}",
        )
        if ok:
            return True, f"enregistré ({len(bets)} paris au total)"
        # A conflict means another write landed first; re-read and try again.
        if message != "conflit" or attempt == WRITE_ATTEMPTS - 1:
            return False, message
    return False, "conflit persistant après plusieurs tentatives"


def ledger_frame(config: GitHubConfig) -> tuple[pd.DataFrame, str]:
    bets, _, message = read_ledger(config)
    if not bets:
        return pd.DataFrame(), message
    return pd.DataFrame(bets), message
