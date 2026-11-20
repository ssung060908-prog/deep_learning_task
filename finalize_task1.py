"""Finalize the frozen stage-5 choice without tuning on holdout."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import experiment as e
import first_version as core

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def checkpoint(model, prep, config, metadata):
    return dict(format_version=1, input_dim=116, config=config,
                preprocessor=prep.state(), metadata=metadata,
                state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default=str(ROOT/'data/heloc_release.csv'))
    p.add_argument('--output', default=str(ROOT/'runs/stage6_final'))
    args=p.parse_args()
    plan=read(ROOT/'configs/stage5_finalization_plan.json')
    protocol, frame, columns, y, dev, holdout, _=e.load_protocol(ROOT/'runs/protocol_v1/protocol.json',args.data)
    assert plan['protocol_id']==protocol['protocol_id'] and plan['data_sha256']==e.sha(args.data)
    assert plan['core_code_sha256']==e.sha(core.__file__)
    assert plan['versions']=={k:importlib.metadata.version(k) for k in plan['versions']}
    config=e.validate_config(plan['config'])
    assert config==read(ROOT/'configs/stage5_selected.json')
    for seed, folder in [('42','stage4_tuning_seed42'),('123','stage5_seed123'),('2026','stage5_seed2026')]:
        assert e.sha(ROOT/'runs'/folder/'search.json')==plan['source_search_sha256'][seed]
    assert plan['epochs']==37 and plan['model_seed']==42
    out=e.empty_output(args.output)
    e.save(out/'frozen_plan.json',plan)
    torch.set_num_threads(4)
    metadata=dict(config=config,epochs=plan['epochs'],seed=plan['model_seed'],
                  data_sha256=plan['data_sha256'],protocol_id=plan['protocol_id'],
                  selected_cv_auroc=plan['mean_cv_auroc'],seed_std=plan['seed_std'],
                  versions=plan['versions'],device='cpu',threads=4,
                  holdout_already_observed=True, private_test_auroc=None)
    def fit(rows):
        prep=core.Preprocessor(columns,config['scaling'],config['interactions'],config['domain_features']).fit(frame.iloc[rows])
        x=prep.transform(frame.iloc[rows])
        assert x.shape[1]==116
        model,_,epochs,history=core.fit_model(x,y[rows],config,plan['epochs'],25,plan['model_seed'],'cpu',valid=None)
        assert epochs==37 and len(history)==37
        return prep,model,history
    print('Training development-only evaluation model: 6316 rows, 37 epochs.',flush=True)
    prep,model,history=fit(dev)
    pred=core.probabilities(model,prep.transform(frame.iloc[holdout]),'cpu')
    score=float(roc_auc_score(y[holdout],pred))
    devmeta=dict(metadata,role='development_only_evaluation',rows=len(dev),holdout_rows=len(holdout),holdout_auroc=score)
    torch.save(checkpoint(model,prep,config,devmeta),out/'development_only.pt')
    np.testing.assert_array_equal(pred,core.predict_proba(frame.iloc[holdout],out/'development_only.pt'))
    pd.DataFrame({'sample_id':frame.iloc[holdout][core.ID].to_numpy(),'is_at_risk':y[holdout],'probability':pred}).to_csv(out/'holdout_predictions.csv',index=False)
    e.save(out/'development_history.json',history)
    e.save(out/'evaluation.json',devmeta)
    print(f'Fixed-plan holdout AUROC={score:.8f}. No settings changed. Refitting all rows.',flush=True)
    prep,model,history=fit(np.arange(len(y)))
    metadata.update(role='all_data_submission',rows=len(y),holdout_rows=len(holdout),holdout_auroc=score,
                    note='Holdout score belongs to the separate development-only model. Holdout was observed historically. Final model uses all supplied rows; private Test performance is unknown.')
    torch.save(checkpoint(model,prep,config,metadata),out/'task1.pt')
    expected=core.probabilities(model,prep.transform(frame),'cpu')
    actual=core.predict_proba(frame,out/'task1.pt')
    np.testing.assert_array_equal(expected,actual)
    np.testing.assert_array_equal(actual,core.predict_proba(frame[columns[::-1]],out/'task1.pt'))
    assert actual.shape==(len(frame),) and np.isfinite(actual).all() and ((actual>=0)&(actual<=1)).all()
    assert len(core.predict_proba(frame.iloc[:0],out/'task1.pt'))==0
    try:
        core.predict_proba(frame.drop(columns=columns[0]),out/'task1.pt')
    except ValueError:
        pass
    else:
        raise AssertionError('Missing input must fail')
    e.save(out/'metrics.json',metadata)
    e.save(out/'final_history.json',history)
    e.save(out/'validation.json',dict(reload_exact=True,column_order_invariant=True,positive_probability_range=True,
           missing_column_rejected=True,empty_input_supported=True,rows_checked=len(frame),
           probability_min=float(actual.min()),probability_max=float(actual.max()),
           weights_sha256=e.sha(out/'task1.pt'),parameter_count=sum(v.numel() for v in model.parameters())))
    (out/'code').mkdir()
    for name in ['first_version.py','experiment.py','finalize_task1.py']:
        shutil.copy2(ROOT/name,out/'code'/name)
    e.load_protocol(ROOT/'runs/protocol_v1/protocol.json',args.data)
    for name,digest in read(ROOT/'models/task1-v1/SHA256SUMS.json').items():
        assert e.sha(ROOT/'models/task1-v1'/name)==digest
    print('Final weights saved; reload, input columns, probabilities, baseline preservation verified.',flush=True)

if __name__=='__main__':
    main()
