"""Check completed models, numerical diagnostics, GIFs, and gallery references."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from experiment import ROOT, load_data, build_model
from train import evaluate


def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    reports=[]; cache={}
    for run in sorted((ROOT/'runs').glob('*')):
        if not (run/'complete.json').exists(): continue
        config=json.loads((run/'config.json').read_text())
        summary=json.loads((run/'analysis/summary.json').read_text())
        key=(config['dataset'],config['seed'],config['train_fraction'])
        if key not in cache: cache[key]=load_data(*key)
        data=cache[key]
        assert data['sha256']==config['dataset_sha256']
        model=build_model(config).to(args.device).eval()
        ck=torch.load(run/'latest.pt',map_location=args.device,weights_only=False)
        assert ck['step']==config['steps']==summary['final_step']
        model.load_state_dict(ck['model'])
        x=torch.tensor(data['x'],device=args.device)
        y=torch.tensor(data[config['target']],dtype=torch.float32,device=args.device)
        test=torch.tensor(data['test'],device=args.device)
        metrics=evaluate(model,x,y,test,data['mean'],data['std'])
        assert abs(metrics['rmse']-summary['final_metrics']['test_rmse'])<2e-6
        with np.load(run/'analysis/diagnostics.npz') as d:
            steps=d['steps']
            assert steps[0]==0 and steps[-1]==config['steps'] and np.all(np.diff(steps)>0)
            assert len(steps)==len(list((run/'checkpoints').glob('step_*.pt')))
            for k in d.files:
                assert np.isfinite(d[k]).all(),(run.name,k)
            with torch.no_grad():
                pred=model(x[torch.tensor(d['probe_indices'],device=args.device)])*data['std']+data['mean']
            np.testing.assert_allclose(pred.cpu().numpy(),d['prediction'][-1],rtol=1e-5,atol=1e-5)
        gifs=[]
        for name in ('representations','frequencies','causal_ablations'):
            path=ROOT/'animations'/run.name/(name+'.gif')
            with Image.open(path) as image:
                assert image.n_frames==len(steps),(path,image.n_frames,len(steps))
                image.seek(image.n_frames-1); image.load()
                gifs.append(dict(path=str(path.relative_to(ROOT)),frames=image.n_frames,size=image.size))
        reports.append(dict(run=run.name,step=ck['step'],verified_test_rmse=metrics['rmse'],
                            checkpoints=len(steps),gifs=gifs))
        print('PASS',run.name,flush=True)
        del model,x,y,test
    page=ROOT/'animations/index.html'
    assert page.exists()
    for reference in re.findall(r'(?:src|href)="([^"]+)"',page.read_text()):
        if reference.startswith(('http','#')): continue
        assert (page.parent/reference).exists(),reference
    for report in reports:
        assert report['run'] in page.read_text()
    report=dict(runs=reports,total_gifs=sum(len(r['gifs']) for r in reports),
        checks=['Dataset identity and split isolation (load_data)',
                'Final checkpoint and full-test RMSE agree with reported results',
                'All diagnostic arrays finite; final probe predictions match checkpoint',
                'Checkpoint steps ordered; GIF frame count and final frame decode',
                'Gallery links and run entries exist'])
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"Verified {len(reports)} runs and {report['total_gifs']} GIFs",flush=True)


if __name__=='__main__': main()
