"""Isolated kernel paper ledger; recompute before accepting any candidate."""
from functools import partial
from src.app import tennis_strategy_ledger as shared
from src.app.tennis_strategy_ledger import initialise, state, proposed_stake, settle
from src.app import wta_kernel_strategy as engine


def record(path, owner, candidate, now=None, *, root):
    fresh = engine.score_fixture(engine.load_bundle(root), candidate['fixture'], now)
    for field in ['strategy_id', 'event_key', 'model_sha256', 'history_sha256', 'eligible',
                  'selected_side', 'pick', 'odds', 'probability', 'probabilities', 'expected_returns']:
        if candidate.get(field) != fresh[field]: raise ValueError('Calcul ou paquet modifié : recalculer la sélection.')
    return shared.record(path, owner, candidate, now, strategy_id=engine.STRATEGY_ID, validator=engine.validate_fixture)


export_backup = partial(shared.export_backup, strategy_id=engine.STRATEGY_ID, backup_format='wta-kernel-paper-v1')
restore_backup = partial(shared.restore_backup, strategy_id=engine.STRATEGY_ID, backup_format='wta-kernel-paper-v1')
