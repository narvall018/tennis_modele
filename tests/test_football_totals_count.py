import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import gammainc,gammaincinv
from scipy.stats import poisson
from src.backtesting.football_totals_count import INTENSITY_COLUMNS,PoissonOffset,intensity_features,inputs,predict_fold


def test_poisson_over_probability_and_inverse():
    means=np.array([.3,1.,2.5,4.,8.])
    np.testing.assert_allclose(gammainc(3,means),poisson.sf(2,means))
    np.testing.assert_allclose(gammaincinv(3,gammainc(3,means)),means)


def test_intensity_features_use_previous_days_not_current_goals_or_shots():
    rows=[]
    for i,day in enumerate(['2020-01-01','2020-01-02','2020-01-02','2020-01-03']):
        rows.append(dict(match_id=str(i),match_date=day,country='X',home_team='A',away_team='B',
                         home_goals=1.,away_goals=2.,postmatch_home_shots_on_target=3.,postmatch_away_shots_on_target=4.))
    frame=pd.DataFrame(rows)
    a=intensity_features(frame)
    changed=frame.copy()
    changed.loc[[1,2],['home_goals','postmatch_home_shots_on_target']]=9.
    b=intensity_features(changed)
    pd.testing.assert_frame_equal(a.iloc[:3],b.iloc[:3])
    assert a.loc[3,'gf_sum']!=b.loc[3,'gf_sum']
    swapped=frame.rename(columns={'home_team':'away_team','away_team':'home_team','home_goals':'away_goals',
                                 'away_goals':'home_goals','postmatch_home_shots_on_target':'postmatch_away_shots_on_target',
                                 'postmatch_away_shots_on_target':'postmatch_home_shots_on_target'})
    pd.testing.assert_frame_equal(a,intensity_features(swapped))


def test_count_model_reduces_to_offset_when_counts_equal_mean():
    rng=np.random.default_rng(2)
    x=rng.normal(size=(100,3)); means=np.full(100,2.5)
    model=PoissonOffset(.01).fit(x,means,means)
    np.testing.assert_allclose(model.predict_mean(x,means),means,atol=1e-7)


def test_count_and_binary_folds_do_not_learn_evaluation_outcomes():
    root=Path(__file__).resolve().parents[1]
    protocol=json.loads((root/'models/football_totals_count/protocol.json').read_text())
    protocol['min_train']=100
    rng=np.random.default_rng(3);n=180
    frame=pd.DataFrame({'match_date':pd.to_datetime(['2015-01-01']*60+['2016-01-01']*60+['2017-01-01']*60),
                        'match_id':np.arange(n).astype(str),'B365>2.5':1.9,'B365<2.5':1.95,
                        'total_goals':rng.poisson(2.5,n)})
    for column in INTENSITY_COLUMNS:
        frame[column]=rng.normal(size=n)
    changed=frame.copy(); changed.loc[120:,'total_goals']=12
    changed['B365C>2.5']=99;changed['postmatch_home_shots']=999
    np.testing.assert_array_equal(inputs(frame)[0],inputs(changed)[0])
    for candidate in protocol['candidates']:
        a,_=predict_fold(frame,candidate,2017,protocol)
        b,_=predict_fold(changed,candidate,2017,protocol)
        np.testing.assert_array_equal(a['p0'],b['p0'])
