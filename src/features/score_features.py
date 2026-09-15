"""Descripteurs de marge tirés du score jeu par jeu.

La colonne ``Score`` des classeurs Tennis-Data n'est lue nulle part ailleurs
dans ce dépôt: aucun module de ``src/features`` ne la parse. Toutes les
variables existantes — Elo, formes 3/5/10/20, taux par surface, H2H — sont
construites sur le seul bit « a gagné / a perdu ». Un 6-0 6-0 et un 7-6 6-7 7-6
y sont la même ligne.

Un match porte pourtant une vingtaine de jeux. La marge en jeux est un signal
beaucoup moins bruité que l'issue binaire, et elle sépare deux joueurs que le
bilan victoires/défaites confond: celui qui gagne large et celui qui gagne de
justesse. C'est la seule source d'information réellement inexploitée des tables
locales, et c'est ce que ce module met à disposition.

Toutes les statistiques sont strictement antérieures au match décrit. L'état
d'un joueur est lu au début de la journée et n'est mis à jour qu'une fois tous
les matchs de cette journée décrits, comme l'exige la règle temporelle du
protocole phase 4.

Constantes fixées d'avance, jamais ajustées sur un rendement:

``GAME_RATIO_HALFLIFE``
    Demi-vie, en matchs, de la moyenne exponentielle du ratio de jeux.
``MOV_REFERENCE``
    Marge en jeux qui vaut un multiplicateur de 1 dans l'Elo à marge.
``EXPECTED_RATIO_SCALE``
    Pente de la conversion écart d'Elo → ratio de jeux attendu. 800 points
    d'écart valent environ 0,73 de ratio attendu; la valeur est déclarée, pas
    estimée sur les données.
``PRIOR_WEIGHT``
    Poids du prior 0,5 dans les taux rétrécis (tie-breaks, sets décisifs,
    renversements). Empêche un joueur à deux tie-breaks de sortir à 100 %.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes du protocole
# ---------------------------------------------------------------------------
GAME_RATIO_HALFLIFE = 20.0
MOV_REFERENCE = 6.0
EXPECTED_RATIO_SCALE = 800.0
PRIOR_WEIGHT = 10.0
ELO_MOV_K = 24.0
ELO_MOV_START = 1500.0
WORKLOAD_WINDOW_DAYS = 14
WORKLOAD_LONG_WINDOW_DAYS = 56
RETIREMENT_WINDOW_DAYS = 365
COMMON_OPPONENT_LOOKBACK_DAYS = 730
COMMON_OPPONENT_MIN = 1

_EWMA_ALPHA = 1.0 - 0.5 ** (1.0 / GAME_RATIO_HALFLIFE)
_SET_TOKEN = re.compile(r"^(\d+)-(\d+)$")

SCORE_FEATURES: List[str] = [
    # Niveau lu dans la marge plutôt que dans l'issue
    "mov_elo_diff",
    "game_ratio_ewma_diff",
    "game_ratio_surface_diff",
    "game_ratio_career_diff",
    # Écart entre la marge réalisée et celle qu'un Elo attendait
    "margin_residual_diff",
    # Compétences de point serré
    "tiebreak_rate_diff",
    "decider_rate_diff",
    "comeback_rate_diff",
    "blowout_win_share_diff",
    "blowout_loss_share_diff",
    # Charge de travail comptée en jeux et non en matchs
    "games_played_14d_diff",
    "games_played_56d_diff",
    "games_in_event_diff",
    # Fragilité physique
    "retirements_365d_diff",
    "days_since_retirement_diff",
    # Graphe des adversaires communs
    "common_opponent_margin",
    "log_common_opponent_count",
]


# ---------------------------------------------------------------------------
# Lecture du score
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedScore:
    """Décomposition d'une chaîne ``6-4 5-7 7-6`` orientée Player_1."""

    sets: Tuple[Tuple[int, int], ...]
    games_1: int
    games_2: int
    sets_1: int
    sets_2: int
    tiebreaks_1: int
    tiebreaks_2: int
    decider_played: bool
    decider_won_by_1: bool
    lost_first_set: bool

    @property
    def total_games(self) -> int:
        return self.games_1 + self.games_2

    @property
    def game_ratio_1(self) -> float:
        total = self.total_games
        return 0.5 if total == 0 else self.games_1 / total


def parse_score(score: object) -> Optional[ParsedScore]:
    """Décompose un score Tennis-Data. Retourne ``None`` si illisible.

    Le format des tables locales est homogène: uniquement des jetons ``N-M``,
    sans détail de points de tie-break. Un set gagné 7-6 ou perdu 6-7 est donc
    reconnu comme tie-break par le seul couple de jeux.
    """
    if not isinstance(score, str):
        return None
    tokens = score.split()
    if not tokens:
        return None

    sets: List[Tuple[int, int]] = []
    for token in tokens:
        match = _SET_TOKEN.match(token)
        if match is None:
            return None
        sets.append((int(match.group(1)), int(match.group(2))))
    if not sets:
        return None

    games_1 = sum(a for a, _ in sets)
    games_2 = sum(b for _, b in sets)
    sets_1 = sum(1 for a, b in sets if a > b)
    sets_2 = sum(1 for a, b in sets if b > a)
    tiebreaks_1 = sum(1 for a, b in sets if (a, b) == (7, 6))
    tiebreaks_2 = sum(1 for a, b in sets if (a, b) == (6, 7))

    decider_played = len(sets) >= 3 and sets_1 >= 1 and sets_2 >= 1
    last_a, last_b = sets[-1]
    decider_won_by_1 = decider_played and last_a > last_b
    lost_first_set = sets[0][1] > sets[0][0]

    return ParsedScore(
        sets=tuple(sets),
        games_1=games_1,
        games_2=games_2,
        sets_1=sets_1,
        sets_2=sets_2,
        tiebreaks_1=tiebreaks_1,
        tiebreaks_2=tiebreaks_2,
        decider_played=decider_played,
        decider_won_by_1=decider_won_by_1,
        lost_first_set=lost_first_set,
    )


# ---------------------------------------------------------------------------
# État d'un joueur
# ---------------------------------------------------------------------------
def _shrunk_rate(won: float, total: float) -> float:
    """Taux rétréci vers 0,5. Deux tie-breaks gagnés ne valent pas 100 %."""
    return (won + 0.5 * PRIOR_WEIGHT) / (total + PRIOR_WEIGHT)


class _Ewma:
    """Moyenne exponentielle corrigée du biais de démarrage."""

    __slots__ = ("_sum", "_weight")

    def __init__(self) -> None:
        self._sum = 0.0
        self._weight = 0.0

    def push(self, value: float) -> None:
        self._sum = _EWMA_ALPHA * value + (1.0 - _EWMA_ALPHA) * self._sum
        self._weight = _EWMA_ALPHA + (1.0 - _EWMA_ALPHA) * self._weight

    def value(self, default: float = 0.5) -> float:
        return default if self._weight <= 0.0 else self._sum / self._weight


@dataclass
class MarginState:
    """Tout ce qu'un joueur a montré dans la marge, avant le match courant."""

    mov_elo: float = ELO_MOV_START
    ratio: _Ewma = field(default_factory=_Ewma)
    residual: _Ewma = field(default_factory=_Ewma)
    surface_ratio: Dict[str, _Ewma] = field(default_factory=lambda: defaultdict(_Ewma))

    career_games_won: float = 0.0
    career_games_total: float = 0.0

    tiebreaks_won: float = 0.0
    tiebreaks_total: float = 0.0
    deciders_won: float = 0.0
    deciders_total: float = 0.0
    comebacks_won: float = 0.0
    comebacks_total: float = 0.0
    blowout_wins: float = 0.0
    blowout_losses: float = 0.0
    scored_matches: float = 0.0

    game_log: deque = field(default_factory=deque)          # (date, games)
    event_games: Dict[Tuple[str, int], float] = field(default_factory=dict)
    retirements: deque = field(default_factory=deque)        # dates d'abandon subi
    opponents: Dict[str, deque] = field(default_factory=lambda: defaultdict(deque))

    # -- lectures (toutes strictement pré-match) -------------------------
    def expected_ratio(self, opponent_elo: float) -> float:
        gap = self.mov_elo - opponent_elo
        return 1.0 / (1.0 + 10.0 ** (-gap / EXPECTED_RATIO_SCALE))

    def career_ratio(self) -> float:
        if self.career_games_total <= 0.0:
            return 0.5
        return self.career_games_won / self.career_games_total

    def games_in_window(self, ref: date, days: int) -> float:
        cutoff = ref - timedelta(days=days)
        return float(sum(games for day, games in self.game_log if day >= cutoff))

    def games_in_event(self, event: Tuple[str, int]) -> float:
        return self.event_games.get(event, 0.0)

    def retirements_in_window(self, ref: date, days: int) -> float:
        cutoff = ref - timedelta(days=days)
        return float(sum(1 for day in self.retirements if day >= cutoff))

    def days_since_retirement(self, ref: date, cap: float = 1095.0) -> float:
        if not self.retirements:
            return cap
        return min(float((ref - self.retirements[-1]).days), cap)

    def opponent_ratio(self, opponent: str, ref: date) -> Optional[float]:
        entries = self.opponents.get(opponent)
        if not entries:
            return None
        cutoff = ref - timedelta(days=COMMON_OPPONENT_LOOKBACK_DAYS)
        kept = [ratio for day, ratio in entries if day >= cutoff]
        if not kept:
            return None
        return float(np.mean(kept))

    # -- écriture (après que tous les matchs du jour ont été décrits) ----
    def prune(self, ref: date) -> None:
        cutoff = ref - timedelta(days=WORKLOAD_LONG_WINDOW_DAYS)
        while self.game_log and self.game_log[0][0] < cutoff:
            self.game_log.popleft()
        retire_cutoff = ref - timedelta(days=RETIREMENT_WINDOW_DAYS)
        while self.retirements and self.retirements[0] < retire_cutoff:
            self.retirements.popleft()


def _update_state(
    state: MarginState,
    *,
    day: date,
    event: Tuple[str, int],
    games_for: int,
    games_against: int,
    parsed: Optional[ParsedScore],
    mine_is_player_1: bool,
    opponent: str,
    surface: str,
    expected_ratio: float,
    completed: bool,
    retired_here: bool,
) -> None:
    """Applique un match à l'état d'un joueur.

    Les matchs non terminés alimentent la charge de travail et le compteur
    d'abandons, jamais les statistiques de marge: un score interrompu ne dit
    pas qui était le meilleur.
    """
    played = games_for + games_against
    if played > 0:
        state.game_log.append((day, float(played)))
        state.event_games[event] = state.event_games.get(event, 0.0) + float(played)
    if retired_here:
        state.retirements.append(day)
    if not completed or parsed is None or played == 0:
        return

    ratio = games_for / played
    state.ratio.push(ratio)
    state.surface_ratio[surface].push(ratio)
    state.residual.push(ratio - expected_ratio)
    state.career_games_won += games_for
    state.career_games_total += played
    state.scored_matches += 1.0
    state.opponents[opponent].append((day, ratio))

    if mine_is_player_1:
        tb_won, tb_lost = parsed.tiebreaks_1, parsed.tiebreaks_2
        won_decider = parsed.decider_won_by_1
        lost_first = parsed.lost_first_set
        won_match = parsed.sets_1 > parsed.sets_2
    else:
        tb_won, tb_lost = parsed.tiebreaks_2, parsed.tiebreaks_1
        won_decider = parsed.decider_played and not parsed.decider_won_by_1
        lost_first = not parsed.lost_first_set and parsed.sets[0][0] != parsed.sets[0][1]
        won_match = parsed.sets_2 > parsed.sets_1

    state.tiebreaks_won += tb_won
    state.tiebreaks_total += tb_won + tb_lost
    if parsed.decider_played:
        state.deciders_total += 1.0
        state.deciders_won += 1.0 if won_decider else 0.0
    if lost_first:
        state.comebacks_total += 1.0
        state.comebacks_won += 1.0 if won_match else 0.0
    if ratio >= 0.70:
        state.blowout_wins += 1.0
    elif ratio <= 0.30:
        state.blowout_losses += 1.0


def _update_mov_elo(
    state_1: MarginState,
    state_2: MarginState,
    *,
    label: int,
    parsed: Optional[ParsedScore],
    completed: bool,
) -> None:
    """Elo dont le pas dépend de la marge en jeux (principe FiveThirtyEight)."""
    if not completed or parsed is None or parsed.total_games == 0:
        return
    expected_1 = 1.0 / (1.0 + 10.0 ** (-(state_1.mov_elo - state_2.mov_elo) / 400.0))
    margin = abs(parsed.games_1 - parsed.games_2)
    multiplier = math.log1p(margin) / math.log1p(MOV_REFERENCE)
    delta = ELO_MOV_K * multiplier * (label - expected_1)
    state_1.mov_elo += delta
    state_2.mov_elo -= delta


# ---------------------------------------------------------------------------
# Construction du tableau
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = (
    "source_row_id", "Date", "Player_1", "Player_2", "Winner", "Score", "Status",
)


def build_score_features(raw: pd.DataFrame, progress=print) -> Tuple[pd.DataFrame, dict]:
    """Une passe chronologique; rien n'est lu qui ne précède le match décrit.

    L'état de chaque joueur est lu au début de la journée et n'est mis à jour
    qu'une fois tous les matchs de cette journée décrits. Deux matchs joués le
    même jour voient donc exactement le même état, ce qui interdit à l'ordre de
    passage à l'intérieur d'une journée de porter de l'information.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Colonnes requises absentes: {missing}")

    frame = raw.copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date", kind="mergesort")
    frame["sf_surface"] = frame.get("Surface", pd.Series("Hard", index=frame.index)).fillna("Hard").astype(str)
    frame["sf_tournament"] = frame.get("Tournament", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["sf_status"] = frame["Status"].fillna("completed").astype(str).str.strip().str.lower()

    states: Dict[str, MarginState] = defaultdict(MarginState)
    rows: List[dict] = []
    audit = {
        "matches_seen": 0,
        "score_unreadable": 0,
        "score_contradicts_winner": 0,
        "non_completed": 0,
    }

    for day_ts, day_rows in frame.groupby("Date", sort=True):
        day = day_ts.date()
        pending: List[tuple] = []

        for row in day_rows.itertuples(index=False):
            audit["matches_seen"] += 1
            p1, p2 = str(row.Player_1), str(row.Player_2)
            winner = str(row.Winner)
            label = 1 if winner == p1 else 0
            surface = row.sf_surface
            event = (row.sf_tournament, day.year)
            status = row.sf_status
            completed = status == "completed"
            if not completed:
                audit["non_completed"] += 1

            parsed = parse_score(row.Score)
            if parsed is None:
                audit["score_unreadable"] += 1
            elif completed and (parsed.sets_1 > parsed.sets_2) != (label == 1):
                # Le score contredit la colonne vainqueur: ligne source fautive,
                # écartée des statistiques de marge plutôt que devinée.
                audit["score_contradicts_winner"] += 1
                parsed = None

            s1, s2 = states[p1], states[p2]

            # --- lecture: tout ce qui suit précède strictement le match ---
            record = {
                "source_row_id": int(row.source_row_id),
                "mov_elo_diff": s1.mov_elo - s2.mov_elo,
                "game_ratio_ewma_diff": s1.ratio.value() - s2.ratio.value(),
                "game_ratio_surface_diff": (
                    s1.surface_ratio[surface].value() - s2.surface_ratio[surface].value()
                ),
                "game_ratio_career_diff": s1.career_ratio() - s2.career_ratio(),
                "margin_residual_diff": s1.residual.value(0.0) - s2.residual.value(0.0),
                "tiebreak_rate_diff": (
                    _shrunk_rate(s1.tiebreaks_won, s1.tiebreaks_total)
                    - _shrunk_rate(s2.tiebreaks_won, s2.tiebreaks_total)
                ),
                "decider_rate_diff": (
                    _shrunk_rate(s1.deciders_won, s1.deciders_total)
                    - _shrunk_rate(s2.deciders_won, s2.deciders_total)
                ),
                "comeback_rate_diff": (
                    _shrunk_rate(s1.comebacks_won, s1.comebacks_total)
                    - _shrunk_rate(s2.comebacks_won, s2.comebacks_total)
                ),
                "blowout_win_share_diff": (
                    s1.blowout_wins / max(s1.scored_matches, 1.0)
                    - s2.blowout_wins / max(s2.scored_matches, 1.0)
                ),
                "blowout_loss_share_diff": (
                    s1.blowout_losses / max(s1.scored_matches, 1.0)
                    - s2.blowout_losses / max(s2.scored_matches, 1.0)
                ),
                "games_played_14d_diff": (
                    s1.games_in_window(day, WORKLOAD_WINDOW_DAYS)
                    - s2.games_in_window(day, WORKLOAD_WINDOW_DAYS)
                ),
                "games_played_56d_diff": (
                    s1.games_in_window(day, WORKLOAD_LONG_WINDOW_DAYS)
                    - s2.games_in_window(day, WORKLOAD_LONG_WINDOW_DAYS)
                ),
                "games_in_event_diff": s1.games_in_event(event) - s2.games_in_event(event),
                "retirements_365d_diff": (
                    s1.retirements_in_window(day, RETIREMENT_WINDOW_DAYS)
                    - s2.retirements_in_window(day, RETIREMENT_WINDOW_DAYS)
                ),
                "days_since_retirement_diff": (
                    s1.days_since_retirement(day) - s2.days_since_retirement(day)
                ),
            }
            margin, count = _common_opponent_signal(s1, s2, day)
            record["common_opponent_margin"] = margin
            record["log_common_opponent_count"] = math.log1p(count)
            rows.append(record)

            pending.append((p1, p2, s1, s2, parsed, label, surface, event, completed, status))

        # --- écriture: seulement après que la journée entière a été décrite ---
        for p1, p2, s1, s2, parsed, label, surface, event, completed, status in pending:
            retired = status in {"retired", "walkover", "defaulted"}
            games_1 = parsed.games_1 if parsed is not None else 0
            games_2 = parsed.games_2 if parsed is not None else 0
            expected_1 = s1.expected_ratio(s2.mov_elo)
            _update_state(
                s1, day=day, event=event, games_for=games_1, games_against=games_2,
                parsed=parsed, mine_is_player_1=True, opponent=p2, surface=surface,
                expected_ratio=expected_1, completed=completed,
                retired_here=retired and label == 0,
            )
            _update_state(
                s2, day=day, event=event, games_for=games_2, games_against=games_1,
                parsed=parsed, mine_is_player_1=False, opponent=p1, surface=surface,
                expected_ratio=1.0 - expected_1, completed=completed,
                retired_here=retired and label == 1,
            )
            _update_mov_elo(s1, s2, label=label, parsed=parsed, completed=completed)
            s1.prune(day)
            s2.prune(day)

    out = pd.DataFrame(rows)
    audit["players_tracked"] = len(states)
    audit["features"] = list(SCORE_FEATURES)
    progress(
        f"Descripteurs de marge: {len(out):,} matchs, {len(states):,} joueurs, "
        f"{audit['score_unreadable']} scores illisibles, "
        f"{audit['score_contradicts_winner']} scores contredisant le vainqueur"
    )
    return out, audit


def _common_opponent_signal(
    state_1: MarginState, state_2: MarginState, day: date
) -> Tuple[float, int]:
    """Écart de marge des deux joueurs contre les adversaires qu'ils partagent.

    Comparer deux joueurs qui ne se sont jamais rencontrés passe par ceux qu'ils
    ont tous deux affrontés. Le H2H existant ne répond qu'au cas où ils se sont
    déjà joués; ce signal-là couvre les autres.
    """
    if not state_1.opponents or not state_2.opponents:
        return 0.0, 0
    shared = state_1.opponents.keys() & state_2.opponents.keys()
    if not shared:
        return 0.0, 0
    gaps: List[float] = []
    for opponent in shared:
        r1 = state_1.opponent_ratio(opponent, day)
        if r1 is None:
            continue
        r2 = state_2.opponent_ratio(opponent, day)
        if r2 is None:
            continue
        gaps.append(r1 - r2)
    if len(gaps) < COMMON_OPPONENT_MIN:
        return 0.0, 0
    return float(np.mean(gaps)), len(gaps)
