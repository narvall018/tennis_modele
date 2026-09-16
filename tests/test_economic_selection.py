from copy import deepcopy
from src.backtesting.economic_selection import admitted,choose,gate


def score(name,lower,roi,n=100):
    return {'model':name,'threshold':.02,'summary':{'settled':n,'roi':{'0.02':roi,'0.05':roi-.01},
        'uncertainty':{'ci95':[lower,.5],'family_lower':lower-.05},'roi_without_best_year':roi-.02,
        'yearly':{'2023':{'roi':roi},'2024':{'roi':roi},'2025':{'roi':roi}}}}


def test_economic_choice_is_lower_bound_not_maximum_profit_or_log_loss():
    scores=[score('high_roi',-.2,.5),score('better_lower',-.01,.1),score('too_few',.4,.6,10)]
    assert choose(scores)['model']=='better_lower'
    assert not admitted(choose(scores)['summary'])


def test_no_eligible_pair_means_abstention_and_ties_are_stable():
    assert choose([score('sparse',.1,.2,49)]) is None
    scores=[score('first',.1,.2),score('second',.1,.3)]
    assert choose(scores)['model']=='first'


def test_evaluation_fields_cannot_change_selection():
    scores=[score('a',.01,.1),score('b',.02,.08)]
    altered=deepcopy(scores)
    altered[0]['evaluation_roi']=100
    altered[1]['evaluation_roi']=-1
    assert choose(altered)['model']==choose(scores)['model']


def test_positive_evaluation_does_not_override_failed_tuning_or_uncertainty():
    summary=score('x',.1,.2,300)['summary']
    assert gate(summary,True)
    assert not gate(summary,False)
    summary['uncertainty']['family_lower']=-.001
    assert not gate(summary,True)
