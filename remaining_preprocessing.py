"""Finish v2 preprocessing comparisons without evaluating holdout."""
from pathlib import Path
import json,importlib.metadata,shutil
import numpy as np
import torch
import first_version as core
import experiment as e
ROOT=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
class Clip16(core.Preprocessor):
    def transform(self,frame):
        values,flags,missing=self.clean(frame)
        values=np.where(np.isnan(values),self.median,values)
        values=np.clip((values-self.center)/self.scale,-16,16)
        result=np.concatenate([values,flags,missing.mean(axis=1,keepdims=True)],axis=1).astype(np.float32)
        if not np.isfinite(result).all(): raise ValueError('Nonfinite values')
        return result
def main():
    p,frame,columns,y,dev,holdout,folds=e.load_protocol(ROOT/'runs/protocol_v1/protocol.json',ROOT/'data/heloc_release.csv')
    config=read(ROOT/'configs/stage5_selected.json')
    plan=read(ROOT/'configs/stage5_finalization_plan.json')
    assert config==plan['config'] and e.sha(core.__file__)==plan['core_code_sha256']
    versions={k:importlib.metadata.version(k) for k in plan['versions']}
    assert versions==plan['versions']
    baseline={}
    for seed,folder in [(42,'stage4_tuning_seed42'),(123,'stage5_seed123'),(2026,'stage5_seed2026')]:
        source=ROOT/'runs'/folder
        assert e.sha(source/'search.json')==plan['source_search_sha256'][str(seed)]
        meta=read(source/'run.json')
        assert all(meta[k]==v for k,v in dict(protocol_id=p['protocol_id'],core_code_sha256=e.sha(core.__file__),versions=versions,epochs=200,patience=25,threads=4,device='cpu',model_seed=seed).items())
        baseline[seed]=next(t for t in read(source/'search.json') if t['config']==config)
    out=e.empty_output(ROOT/'experiments/post_v2_remaining')
    e.save(out/'design.json',dict(date='2026-11-22',config=config,baseline=baseline,protocol_id=p['protocol_id'],
          core_sha256=e.sha(core.__file__),code_sha256=e.sha(__file__),versions=versions,
          screening='Clip16: seed42 first; if mean exceeds same-seed v2, confirm seeds123 and2026. No automatic replacement or holdout evaluation.',
          missing_check='Check normalized missing values against zero: robust center equals fitted median. Reuse identical representation instead of retraining.'))
    checks=[];prepared=[]
    for tr,va in folds:
        assert not set(tr)&set(holdout) and not set(va)&set(holdout)
        prep=core.Preprocessor(columns,'robust',False,False).fit(frame.iloc[tr])
        np.testing.assert_array_equal(prep.center,prep.median)
        for rows in [tr,va]:
            x=prep.transform(frame.iloc[rows]);_,_,missing=prep.clean(frame.iloc[rows])
            masked=x.copy();masked[:,:len(columns)]=np.where(missing,0,masked[:,:len(columns)])
            np.testing.assert_array_equal(x,masked)
        q=Clip16(columns,'robust',False,False).fit(frame.iloc[tr])
        x=q.transform(frame.iloc[tr]);v=q.transform(frame.iloc[va])
        old=prep.transform(frame.iloc[va])
        np.testing.assert_array_equal(old[:,23:],v[:,23:])
        np.testing.assert_array_equal(old[:,:23],np.clip(v[:,:23],-8,8))
        np.testing.assert_array_equal(v,q.transform(frame.iloc[va][columns[::-1]]))
        checks.append(dict(zero_mask_exact=True,changed_validation_numeric_values=int(np.count_nonzero(v[:,:23]!=old[:,:23]))))
        prepared.append((x,y[tr],v,y[va]))
    e.save(out/'preprocessing_checks.json',checks)
    torch.set_num_threads(4)
    results=[]
    for seed in [42,123,2026]:
        scores=[];epochs=[];histories=[]
        for i,(x,yt,v,yv) in enumerate(prepared):
            _,score,epoch,h=core.fit_model(x,yt,config,200,25,seed+i,'cpu',(v,yv))
            if len(h)==200 and 200-epoch<25:
                _,score,epoch,h=core.fit_model(x,yt,config,500,25,seed+i,'cpu',(v,yv))
                assert not (len(h)==500 and 500-epoch<25),'Unresolved capped fold'
            scores.append(float(score));epochs.append(int(epoch));histories.append(h)
            print(f'clip16 seed{seed} fold{i+1}: AUROC={score:.8f}, epoch={epoch}',flush=True)
        result=dict(seed=seed,scores=scores,epochs=epochs,mean_auroc=float(np.mean(scores)),
                    delta=float(np.mean(scores)-baseline[seed]['mean_auroc']))
        results.append(result);e.save(out/f'history_seed{seed}.json',histories);e.save(out/'results.json',results)
        if seed==42 and result['delta']<=0: break
    summary=dict(results=results,baseline_seed42=baseline[42]['mean_auroc'],
                 baseline_matched_mean=float(np.mean([baseline[r['seed']]['mean_auroc'] for r in results])),
                 candidate_mean=float(np.mean([r['mean_auroc'] for r in results])),
                 zero_mask_identical=True,holdout_evaluated=False,new_weights=False)
    e.save(out/'summary.json',summary)
    for folder in [ROOT/'models/task1-v1',ROOT/'models/task1-v2']:
        for name,digest in read(folder/'SHA256SUMS.json').items(): assert e.sha(folder/name)==digest
    print(json.dumps(summary),flush=True)
if __name__=='__main__': main()
