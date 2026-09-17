"""Separate reference-price paper bankroll, never mixed with old ATP or WTA."""
from functools import partial

from src.app import tennis_strategy_ledger as shared
from src.app.tennis_strategy_ledger import initialise, state, proposed_stake, settle
from src.app.atp_reference_strategy import STRATEGY_ID, validate_fixture, verify_candidate


def record(path, owner, candidate, now=None):
    verify_candidate(candidate, now)
    return shared.record(path, owner, candidate, now, strategy_id=STRATEGY_ID, validator=validate_fixture)


export_backup = partial(shared.export_backup, strategy_id=STRATEGY_ID, backup_format='atp-reference-paper-v1')
restore_backup = partial(shared.restore_backup, strategy_id=STRATEGY_ID, backup_format='atp-reference-paper-v1')
