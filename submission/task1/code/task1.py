"""Reproduce the final single DNN on all provided HELOC rows (no search)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import first_version as core

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,help='Original provided training CSV')
    parser.add_argument('--output',default='retrained_task1.pt')
    args=parser.parse_args()
    out=Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    plan=json.loads(Path(__file__).with_name('finalization_plan.json').read_text(encoding='utf-8'))
    if hashlib.sha256(Path(args.data).read_bytes()).hexdigest()!=plan['data_sha256']:
        raise ValueError('Use the original provided training CSV with unchanged row order')
    frame,columns,y=core.load_training(args.data)
    config=plan['config']
    torch.set_num_threads(4)
    prep=core.Preprocessor(columns,config['scaling'],config['interactions'],config['domain_features']).fit(frame)
    x=prep.transform(frame)
    model,_,_,_=core.fit_model(x,y,config,plan['epochs'],25,plan['model_seed'],'cpu',valid=None)
    torch.save(dict(format_version=1,input_dim=x.shape[1],config=config,preprocessor=prep.state(),
                    metadata=dict(role='all_data_submission',rows=len(y),epochs=plan['epochs'],seed=plan['model_seed']),
                    state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),out)
    np.testing.assert_array_equal(core.probabilities(model,x,'cpu'),core.predict_proba(frame,out))
    print(f'Saved and reload-verified: {out}')

if __name__=='__main__':
    main()
