from dataclasses import replace

import numpy as np
import pytest

from trajectorylab import bidding_delayed_strategy as s
from trajectorylab.bidding_delayed_experiment import PolicyView, feedback_from_view
from trajectorylab.bidding_pending_response import Cohort, EmpiricalDelayCDF, PendingEstimator


@pytest.fixture
def config():
    return s.base.load_config('configs/bidding-delayed-strategy.toml')


def test_twelve_configs_and_fixed_equivalence(config):
    assert len(s.variants())==12
    assert len({v['name'] for v in s.variants()})==12
    actions=np.asarray(config['study']['actions'])
    view=PolicyView(24,90.,4,300.,10.,())
    state=dict(pending=3.,log_scale=0.,prior_log_scale=0.,over_probability=0.,information=0.)
    for variant in s.variants()[:2]:
        assert s.policy_action(view,actions,variant,state,config['strategy'])==feedback_from_view(view,actions,4+variant['pending']*3,.3,8)
    state['pending']=0
    for variant in s.variants():
        assert s.policy_action(view,actions,variant,state,config['strategy'])==feedback_from_view(view,actions,4,.3,8)


def test_single_distribution_call_and_startup_neutral(config):
    est=PendingEstimator(EmpiricalDelayCDF.fit([0,2],2))
    view=PolicyView(0,0.,0,100.,10.,())
    calls=[]
    original=est._distribution
    def tracked(cohorts):
        calls.append(1)
        return original(cohorts)
    est._distribution=tracked
    state=s.posterior_state(view,est)
    assert calls==[1]
    assert state['log_scale']==state['prior_log_scale']
    a=np.asarray(config['study']['actions'])
    for v in s.variants():
        assert s.policy_action(view,a,v,state,config['strategy'])==feedback_from_view(view,a,0,.3,8)


def test_partial_gate_direction_bounds_and_absolute_difference(config):
    view=PolicyView(24,80.,2,200.,10.,())
    settings=config['strategy']
    actions=np.linspace(.05,3,1000)
    state=dict(pending=20.,log_scale=np.log(.5),prior_log_scale=np.log(.8),over_probability=.95,information=8.)
    variants=s.variants()
    values=[s.policy_action(view,actions,v,state,settings) for v in variants[:4]]
    assert values[0]<=values[2]<=values[3]<=values[1]
    assert s.policy_action(view,actions,variants[4],state,settings)==values[2]
    low=dict(state,information=7.99)
    assert s.policy_action(view,actions,variants[4],low,settings)==values[1]
    high=dict(state,log_scale=np.log(1.5),over_probability=.01)
    assert s.policy_action(view,actions,variants[4],high,settings)==values[1]
    assert s.policy_action(view,actions,variants[7],high,settings)>values[1]
    # Same absolute k=1 has relative-prior displacement but absolute factor one.
    neutral=dict(state,log_scale=0.)
    assert s.policy_action(view,actions,variants[7],neutral,settings)>s.policy_action(view,actions,variants[10],neutral,settings)
    for v in variants:
        assert .05<=s.policy_action(view,actions,v,state,settings)<=3
    assert s.policy_action(replace(view,spent=0,arrived=0),actions,variants[0],state,settings)>0


def synthetic_rows(config, improvement=1.05):
    rows=[]
    for variant in s.variants():
        for market in config['study']['validation_markets']:
            for group in ('under','nominal','over'):
                for delay in config['evaluation']['regimes']:
                    rows.append(dict(method=variant['name'],market=market,bias_group=group,delay=delay,
                                     score=10*(improvement if variant['family']=='partial' else 1.),
                                     conversions=10.,violation=0.,budget_utilization=.5))
    return rows


def test_selection_all_gates_and_deterministic_simpler_tie(config):
    selected=s.development_selection(synthetic_rows(config),config)
    assert selected['gate']
    assert selected['selected']['name']=='partial-0.25'
    partial=next(r for r in selected['table'] if r['variant']['name']=='partial-0.25')
    assert all(partial['gates'].values()) and partial['market_wins']==6
    assert len(partial['over_cells_vs_naive'])==3
    failed=s.development_selection(synthetic_rows(config,1.),config)
    assert not failed['gate'] and failed['selected'] is None
    rows=synthetic_rows(config)
    for r in rows:
        if r['method'].startswith('partial') and r['bias_group']=='over' and r['delay']=='long':
            r['violation']=.04
    assert not s.development_selection(rows,config)['gate']


def test_ten_endpoint_confirmation_statistics(config):
    config={**config,'evaluation':{**config['evaluation'],'bootstrap_draws':100}}
    rows=[]
    for m in range(7101,7125):
        for g in ('under','nominal','over'):
            for name in ['fixed-naive','fixed-posterior','partial-0.25']:
                rows.append(dict(market=m,bias_group=g,method=name,score=10.5 if name=='partial-0.25' else 10.,conversions=10.,violation=0.,budget_utilization=.5))
    selection=dict(selected={'name':'partial-0.25'},stronger_fixed_reference='fixed-posterior')
    result=s.confirmation_statistics(rows,selection,config)
    assert result['passed'] and result['protection_endpoints']==10
    assert result['one_sided_alpha']==.005
    assert len(result['protection'])==4
    assert result['primary']['ci95'][0]==pytest.approx(1.05)


def test_history_formula_adapter_and_config_market_boundary(config):
    view=PolicyView(1,1,1,100,10,(Cohort(0,.2,1,10),))
    est=PendingEstimator(EmpiricalDelayCDF.fit([0,2],2),scales=[1.])
    state=s.posterior_state(view,est)
    assert state['pending']==pytest.approx(1.)
    assert state['information']==pytest.approx(1.)
    assert config['study']['validation_markets']==list(range(2001,2007))
    assert config['study']['confirmation_markets']==list(range(7101,7125))


def test_raw_action_factor_precedes_saturation_and_quantization(config):
    view=PolicyView(47,0.,100,100.,10.,())
    state=dict(pending=0.,log_scale=-10.,prior_log_scale=0.,over_probability=1.,information=100.)
    variant=next(v for v in s.variants() if v['name']=='centered-eta1-pending0')
    assert s.policy_action(view,np.asarray(config['study']['actions']),variant,state,config['strategy'])==3.
    # Premature raw-action clipping would yield 3*(2/3)=2, a different action.
    assert s.policy_action(view,np.asarray(config['study']['actions']),variant,state,config['strategy'])!=2.


def test_float32_reference_equivalence_and_tie_lower(config):
    rng=np.random.default_rng(4)
    actions=np.asarray(config['study']['actions'])
    state=dict(pending=3.14159,log_scale=0.,prior_log_scale=0.,over_probability=0.,information=0.)
    for _ in range(100):
        view=PolicyView(int(rng.integers(48)),float(rng.uniform(0,100)),int(rng.integers(10)),100.,10.,())
        for variant in s.variants()[:2]:
            expected=feedback_from_view(view,actions,view.arrived+variant['pending']*state['pending'],.3,8)
            assert s.policy_action(view,actions,variant,state,config['strategy'])==expected
    view=PolicyView(0,0,0,100,10,())
    assert s.policy_action(view,np.array([.75,1.25]),s.variants()[0],state,config['strategy'])==.75


def test_freeze_rejects_changed_payload(tmp_path):
    path=tmp_path/'record.json'
    s.freeze(path,{'identity':'first'})
    s.freeze(path,{'identity':'first'})
    with pytest.raises(RuntimeError,match='Frozen artifact'):
        s.freeze(path,{'identity':'other'})


def test_confirm_false_gate_blocks_before_fresh_market_access(tmp_path,config,monkeypatch):
    input_dir,output=tmp_path/'input',tmp_path/'extension'
    core={'source':s.base.source_record()}
    s.write_json(input_dir/'history.json',{})
    history_hash=s.file_hash(input_dir/'history.json')
    identity=s.fingerprint(core)
    s.write_json(input_dir/'history-manifest.json',{'identity':identity,'core':core,'history_sha256':history_hash})
    config_path='configs/bidding-delayed-strategy.toml'
    provenance=dict(config=config,source=s.source_record(config_path),history_sha256=history_hash,training_history_identity=identity)
    selected=dict(gate=False,provenance=provenance)
    selected['identity']=s.fingerprint(selected)
    s.write_json(output/'selection.json',selected)
    calls=[]
    monkeypatch.setattr(s,'build_event_market',lambda *args:calls.append(args))
    with pytest.raises(RuntimeError,match='passing selection'):
        s.run(config_path,input_dir,output,'confirm')
    assert calls==[]
    assert not (output/'confirmation').exists()
