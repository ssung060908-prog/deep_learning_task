"""CV-only test of non-neural piecewise-linear numerical encoding."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import warnings
import numpy as np
import torch
import first_version as core
import experiment as e

ROOT=Path(__file__).resolve().parent

class PLEPreprocessor:
    def __init__(self,columns,bins):
        if type(bins) is not int or bins<2:
            raise ValueError('bins must be an integer >=2')
        self.columns=list(columns)
        self.bins=bins
        self.cleaner=core.Preprocessor(columns,domain_features=False)

    def fit(self,frame):
        values,_,_=self.cleaner.clean(frame)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.median=np.nanmedian(values,axis=0)
        self.median=np.nan_to_num(self.median,nan=0.)
        self.edges=[]
        for j in range(len(self.columns)):
            observed=values[:,j][np.isfinite(values[:,j])]
            # Quantiles use observed training values only, never validation or imputed copies.
            self.edges.append(np.unique(np.quantile(observed,np.linspace(0,1,self.bins+1)))
                              if len(observed) else np.array([0.]))
        return self

    def transform(self,frame):
        values,flags,missing=self.cleaner.clean(frame)
        values=np.where(np.isnan(values),self.median,values)
        parts=[]
        for j,edges in enumerate(self.edges):
            if len(edges)<2:
                parts.append(np.zeros((len(frame),1)))
            else:
                parts.append(np.clip((values[:,j,None]-edges[:-1])/np.diff(edges),0.,1.))
        result=np.concatenate(parts+[flags,missing.mean(axis=1,keepdims=True)],axis=1).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite encoded input')
        return result

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def run(args):
    protocol,frame,columns,y,dev,holdout,folds=e.load_protocol(ROOT/'runs/protocol_v1/protocol.json',ROOT/'data/heloc_release.csv')
    config=e.validate_config(read(ROOT/'configs/stage5_selected.json'))
    source=ROOT/'runs/stage4_tuning_seed42'
    prior=read(source/'run.json')
    versions={k:importlib.metadata.version(k) for k in ('numpy','pandas','scikit-learn','torch')}
    assert prior['protocol_id']==protocol['protocol_id'] and prior['core_code_sha256']==e.sha(core.__file__)
    assert prior['versions']==versions and prior['model_seed']==42 and prior['epochs']==200 and prior['patience']==25
    assert prior['threads']==4 and prior['device']=='cpu'
    baseline=next(t for t in read(source/'search.json') if t['config']==config)
    out=e.empty_output(args.output)
    design=dict(date='2026-11-20',mode='cv_only',bins=[8,16],config=config,model_seed=42,
                epochs=200,patience=25,threads=4,device='cpu',protocol_id=protocol['protocol_id'],
                data_sha256=protocol['data_sha256'],versions=versions,core_code_sha256=e.sha(core.__file__),
                code_sha256=e.sha(__file__),source_search_sha256=e.sha(source/'search.json'),
                baseline=baseline,holdout_evaluated=False,
                preprocessing='Replace robust-scaled numeric values with clipped cumulative PLE; retain same special-code flags and missing fraction. Observed training quantiles; unique boundaries; constant columns zero.',
                decision_rule='Screen only: positive seed42 mean difference is a candidate for later multiseed confirmation, never automatic v2 replacement.')
    e.save(out/'design.json',design)
    torch.set_num_threads(4)
    results=[]
    for bins in [8,16]:
        scores=[];epochs=[];histories=[];dims=[];fold_info=[]
        for number,(tr,va) in enumerate(folds):
            assert not set(tr)&set(holdout) and not set(va)&set(holdout)
            prep=PLEPreprocessor(columns,bins).fit(frame.iloc[tr])
            x=prep.transform(frame.iloc[tr]);v=prep.transform(frame.iloc[va])
            _,score,epoch,history=core.fit_model(x,y[tr],config,200,25,42+number,'cpu',(v,y[va]))
            scores.append(float(score));epochs.append(int(epoch));histories.append(history);dims.append(x.shape[1])
            fold_info.append(dict(input_dim=x.shape[1],actual_intervals=[max(1,len(b)-1) for b in prep.edges],
                                  parameter_count=(x.shape[1]+1)*32+33,
                                  capped=len(history)==200 and 200-epoch<25))
            print(f'PLE {bins} bins, fold {number+1}/5: AUROC={score:.8f}, best epoch={epoch}, dimensions={x.shape[1]}',flush=True)
            e.save(out/f'ple{bins}_partial.json',dict(scores=scores,epochs=epochs,folds=fold_info))
        result=dict(bins=bins,scores=scores,epochs=epochs,mean_auroc=float(np.mean(scores)),
                    std_auroc=float(np.std(scores)),fold_differences=(np.array(scores)-baseline['scores']).tolist(),
                    delta_vs_v2_seed42=float(np.mean(scores)-baseline['mean_auroc']),folds=fold_info)
        results.append(result)
        e.save(out/f'ple{bins}_history.json',histories)
        e.save(out/'results.json',results)
    e.load_protocol(ROOT/'runs/protocol_v1/protocol.json',ROOT/'data/heloc_release.csv')
    for folder in [ROOT/'models/task1-v1',ROOT/'models/task1-v2']:
        for name,digest in read(folder/'SHA256SUMS.json').items():
            assert e.sha(folder/name)==digest
    shutil.copy2(__file__,out/'ple_experiment.py')
    print('Finished 10 CV fits. No holdout evaluation or new submission weights.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'runs/post_v2_ple_seed42'))
    run(parser.parse_args())
