"""Run the data and model pipelines from inside the app.

Each task is a script this repository already has, invoked as a subprocess with
**the interpreter currently running the app**. That detail is the whole point:
the models were first trained under Python 3.12 / pandas 3.0 while the app runs
under 3.9 / pandas 2.2, and the resulting artefacts would not load. Using
``sys.executable`` makes that mismatch impossible to reintroduce by accident.

Tasks are ordered by dependency and each one states what it rewrites, because
several take minutes and one of them spends API quota.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Task:
    key: str
    label: str
    description: str
    command: list[str]
    produces: list[str]
    minutes: str
    working_directory: str = "."
    spends_quota: bool = False


TASKS: list[Task] = [
    Task(
        key="tennis_data",
        label="Données tennis ATP",
        description="Classeurs Tennis-Data et statistiques TennisMyLife, contrôles qualité "
                    "puis publication atomique.",
        command=["scripts/update_tennis_data.py"],
        produces=["data/atp_tennis.csv", "data/processed/atp_matches_enriched.csv.gz"],
        minutes="8–12 min",
    ),
    Task(
        key="tennis_expansion",
        label="Données WTA, Challenger, qualifications",
        description="Les circuits que le pipeline ATP laisse de côté. La WTA porte des "
                    "cotes; Challenger et qualifications n'en portent pas.",
        command=["scripts/update_tennis_expansion.py"],
        produces=["data/wta_tennis.csv", "data/processed/atp_unpriced_matches.csv.gz"],
        minutes="10–15 min",
    ),
    Task(
        key="football_data",
        label="Données football",
        description="22 divisions européennes, 1X2, plus/moins 2,5 et handicap asiatique, "
                    "cotes d'ouverture et de clôture.",
        command=["scripts/update_football_data.py"],
        produces=["data/football/football_matches.csv.gz"],
        minutes="3–6 min",
    ),
    Task(
        key="ufc_data",
        label="Données UFC",
        description="Extraction UFCStats, classements, snapshots de cotes; complète les "
                    "cartes récentes directement sur le site.",
        command=["run_rigorous_pipeline.py", "update-data"],
        produces=["predictor_ufc/data/rigorous/processed/fights.parquet"],
        minutes="4–8 min",
        working_directory="predictor_ufc",
    ),
    Task(
        key="tennis_ratings",
        label="Elo tennis par joueur",
        description="Recalcule les Elo ATP et WTA et les exporte par nom complet, pour "
                    "rapprocher les matchs cotés en direct.",
        command=["scripts/export_tennis_ratings.py"],
        produces=["models/tennis/atp_player_ratings.parquet",
                  "models/tennis/wta_player_ratings.parquet"],
        minutes="4–7 min",
    ),
    Task(
        key="football_model",
        label="Modèle football",
        description="Descripteurs, comparaison de dix familles en walk-forward, puis "
                    "l'état final de chaque équipe.",
        command=["scripts/train_football_model.py"],
        produces=["models/football/football_model.joblib",
                  "models/football/team_states.parquet"],
        minutes="20–40 min",
    ),
    Task(
        key="ufc_model",
        label="Modèle UFC",
        description="Descripteurs purs sans cote, même comparaison de familles, plus "
                    "l'état des combattants au format portable.",
        command=["scripts_train_descriptor_model.py", "--rebuild-features"],
        produces=["models/ufc/ufc_descriptor_model.joblib",
                  "models/ufc/fighter_states.parquet"],
        minutes="10–20 min",
        working_directory="predictor_ufc",
    ),
]

TASKS_BY_KEY = {task.key: task for task in TASKS}


def artefact_status(root: Path, task: Task) -> list[dict[str, Any]]:
    """Freshness of what a task produces, so staleness is visible before running."""
    rows: list[dict[str, Any]] = []
    for relative in task.produces:
        path = root / relative
        if path.exists():
            stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            age_days = (datetime.now(timezone.utc) - stamp).days
            rows.append({
                "Fichier": relative,
                "État": "présent",
                "Mis à jour": stamp.strftime("%Y-%m-%d %H:%M"),
                "Âge (jours)": age_days,
                "Mo": round(path.stat().st_size / 1_048_576, 1),
            })
        else:
            rows.append({
                "Fichier": relative, "État": "absent", "Mis à jour": "—",
                "Âge (jours)": None, "Mo": None,
            })
    return rows


def run_task(root: Path, key: str, timeout_seconds: int = 5400) -> dict[str, Any]:
    """Run one pipeline and return its outcome, never raising into the UI."""
    task = TASKS_BY_KEY.get(key)
    if task is None:
        return {"ok": False, "output": f"tâche inconnue: {key}"}
    working = (root / task.working_directory).resolve()
    command = [sys.executable, *task.command]
    started = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(
            command, cwd=working, capture_output=True, text=True,
            timeout=timeout_seconds, check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False, "task": task.label,
            "output": f"interrompu après {timeout_seconds // 60} minutes",
        }
    except OSError as error:
        return {"ok": False, "task": task.label, "output": f"lancement impossible: {error}"}

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    output = (completed.stdout or "") + (completed.stderr or "")
    return {
        "ok": completed.returncode == 0,
        "task": task.label,
        "returncode": completed.returncode,
        "seconds": round(elapsed, 1),
        "interpreter": sys.executable,
        # Only the tail matters in a UI, and a full pipeline log is very long.
        "output": output[-6000:],
    }
