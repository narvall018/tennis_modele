"""WTA uses a separate database and backup identity, with shared audited accounting."""
from functools import partial
from src.app.tennis_strategy_ledger import initialise, state, proposed_stake, settle
from src.app import tennis_strategy_ledger as shared
from src.app.wta_strategy import STRATEGY_ID, validate_fixture

record = partial(shared.record, strategy_id=STRATEGY_ID, validator=validate_fixture)
export_backup = partial(shared.export_backup, strategy_id=STRATEGY_ID, backup_format='wta-paper-v1')
restore_backup = partial(shared.restore_backup, strategy_id=STRATEGY_ID, backup_format='wta-paper-v1')
