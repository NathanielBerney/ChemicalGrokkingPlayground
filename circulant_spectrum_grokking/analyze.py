"""Checkpoint diagnostics, learned linear controls, and site-mode interventions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from experiment import load_data, build_model, gini, real_mode_basis, rfft_power, mode_ablation_selectivity


def error_metrics(pred, truth):
    delta=pred-truth
    return dict(rmse=float(np.mean(delta**2)**.5),mae=float(np.mean(abs(delta))),
                per_output_rmse=np.mean(delta**2,axis=0)**.5)


def jsonable(obj):
    if isinstance(obj,dict): return {k:jsonable(v) for k,v in obj.items()}
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,(np.integer,np.floating)): return obj.item()
    return obj


def linear_baselines(d,target):
    # First row contains all independent information in a canonical circulant.
    # Fit raw row coefficients; no hand-coded Fourier features.
    design=np.column_stack((d['x'][:,0].astype(np.float64),np.ones(len(d['x']))))
    y=d[target]
    coef=np.linalg.lstsq(design[d['train']],y[d['train']],rcond=None)[0]
    pred=design@coef
    result={}
    for split in ('train','test'):
        ix=d[split]
        result[split]=error_metrics(pred[ix],y[ix])
    result['design_rank']=int(np.linalg.matrix_rank(design[d['train']]))
    result['description']='Least-squares regression on raw first row plus intercept; fitted on train only'
    return result,coef


def fixed_pca(values):
    shape=values.shape
    rows=values.reshape(-1,shape[-1]).astype(np.float64)
    center=rows.mean(0)
    centered=rows-center
    eigenvalues,eigenvectors=np.linalg.eigh(centered.T@centered)
    basis=eigenvectors[:,-2:][:,::-1]
    for j in range(2):
        if basis[np.argmax(abs(basis[:,j])),j]<0: basis[:,j]*=-1
    fraction=float(eigenvalues[-2:].sum()/max(eigenvalues.sum(),1e-25))
    return (centered@basis).reshape(*shape[:-1],2),fraction,basis,center


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--no-render',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    config=json.loads((args.run/'config.json').read_text())
    d=load_data(config['dataset'],config['seed'],config['train_fraction'])
    if config['dataset_sha256']!=d['sha256']: raise ValueError('Dataset has changed')
    saved=np.load(args.run/'split.npz')
    np.testing.assert_array_equal(saved['train'],d['train'])
    np.testing.assert_array_equal(saved['test'],d['test'])
    indices=np.load(args.run/'probe_indices.npy')
    n=config['n']; target=config['target']; mean,std=d['mean'],d['std']
    device=torch.device(args.device)
    model=build_model(config).to(device).eval()
    x=torch.as_tensor(d['x'][indices],device=device)
    truth=d[target][indices]
    checkpoints=sorted((args.run/'checkpoints').glob('step_*.pt'))
    if not checkpoints: raise ValueError('No checkpoints')
    # Paired Fourier coordinates count as one real mode subspace.
    modes=[torch.tensor(real_mode_basis(n,k),dtype=torch.float32,device=device)
           for k in range(1,n//2+1)]
    rng=np.random.default_rng(923)
    random_bases={}
    for rank in (1,2):
        random_bases[rank]=[]
        for _ in range(3):
            q=rng.normal(size=(n,rank)); q-=q.mean(0)
            q=np.linalg.qr(q)[0]
            random_bases[rank].append(torch.tensor(q,dtype=torch.float32,device=device))
    stages=list(range(config['layers']))
    # Symmetric offset coefficients c_1,...,c_{n/2}; diagonal fixed at zero.
    delta=(np.arange(n)[None,:]-np.arange(n)[:,None])%n
    directions=np.stack([((delta==r)|(delta==n-r)).astype(np.float32)
                         for r in range(1,n//2+1)])
    eps=.001
    sensitivity_indices=indices[:4]
    reference=d['x'][sensitivity_indices]
    plus=reference[:,None]+eps*directions[None]
    minus=reference[:,None]-eps*directions[None]
    perturb=torch.tensor(np.concatenate((plus,minus),axis=0).reshape(-1,n,n),device=device)
    def analytic_targets(matrices):
        z=np.fft.fft(matrices[...,0,:].astype(np.float64),axis=-1).real
        return np.sort(z,axis=-1) if target=='sorted' else z
    true_sensitivity=((analytic_targets(plus)-analytic_targets(minus))/(2*eps)).transpose(0,2,1)
    store={k:[] for k in ('steps','prediction','weight','weight_power','hidden_power',
           'row_hidden','cls_hidden','probe_rmse','ablation_rmse','ablation_output_change',
           'random_ablation_rmse','sensitivity','sensitivity_relative_error')}
    with torch.no_grad():
        for path in checkpoints:
            ck=torch.load(path,map_location=device,weights_only=False)
            model.load_state_dict(ck['model'])
            pred,hidden=model(x,return_hidden=True)
            pred_np=(pred*std+mean).cpu().numpy()
            weight=model.row_embed.weight.cpu().numpy().copy()
            # Final stage is displayed but not ablated: it cannot affect CLS afterward.
            powers=np.stack([rfft_power(h[:16,1:].cpu().numpy(),1).mean(0).T for h in hidden])
            rmses=[]; changes=[]; random_rmses=[]
            for stage in stages:
                stage_rmses=[]; stage_changes=[]
                for q in modes:
                    altered=(model(x,intervention=(stage,q))*std+mean).cpu().numpy()
                    stage_rmses.append(float(np.mean((altered-truth)**2)**.5))
                    stage_changes.append(np.mean((altered-pred_np)**2,axis=0)**.5)
                rmses.append(stage_rmses); changes.append(stage_changes)
                stage_random=[]
                for rank in (1,2):
                    scores=[]
                    for q in random_bases[rank]:
                        altered=(model(x,intervention=(stage,q))*std+mean).cpu().numpy()
                        scores.append(float(np.mean((altered-truth)**2)**.5))
                    stage_random.append(scores)
                random_rmses.append(stage_random)
            perturbed=(model(perturb)*std+mean).cpu().numpy().reshape(8,n//2,n)
            sensitivity=((perturbed[:4]-perturbed[4:])/(2*eps)).transpose(0,2,1)
            relative=float(np.linalg.norm(sensitivity-true_sensitivity)/np.linalg.norm(true_sensitivity))
            values=dict(steps=ck['step'],prediction=pred_np,weight=weight,
                weight_power=rfft_power(weight,1),hidden_power=powers,
                row_hidden=hidden[-1][0,1:].cpu().numpy(),
                cls_hidden=hidden[-1][:,0].cpu().numpy(),
                probe_rmse=float(np.mean((pred_np-truth)**2)**.5),
                ablation_rmse=rmses,ablation_output_change=changes,
                random_ablation_rmse=random_rmses,sensitivity=sensitivity,
                sensitivity_relative_error=relative)
            for k,v in values.items(): store[k].append(v)
            print(f"{args.run.name} step={ck['step']} probe_rmse={values['probe_rmse']:.5g} "
                  f"sensitivity_relative_error={relative:.4g}",flush=True)
    arrays={k:np.asarray(v) for k,v in store.items()}
    arrays.update(truth=truth,probe_indices=indices,true_sensitivity=true_sensitivity,
                  sensitivity_indices=sensitivity_indices,stages=np.array(stages),
                  coupling_count=np.count_nonzero(d['x'][indices,0,1:n//2+1],axis=1))
    pca_summary={}
    for name,values in [('weight',arrays['weight'].transpose(0,2,1)),
                        ('row',arrays['row_hidden']),('cls',arrays['cls_hidden'])]:
        projected,fraction,basis,center=fixed_pca(values)
        arrays[name+'_pca']=projected
        arrays[name+'_pca_basis']=basis
        arrays[name+'_pca_center']=center
        pca_summary[name]=fraction
    baseline,coef=linear_baselines(d,target)
    arrays['linear_coefficients']=coef
    for split in ('train','test'):
        ix=d[split]
        base=np.broadcast_to(d[target][d['train']].mean(0),(len(ix),n))
        baseline[split]['mean_predictor_rmse']=error_metrics(base,d[target][ix])['rmse']
    with (args.run/'metrics.csv').open() as f:
        metric_rows=list(csv.DictReader(f))
    summary=dict(run=args.run.name,target=target,seed=config['seed'],
        final_step=int(arrays['steps'][-1]),checkpoints=len(checkpoints),
        train_size=len(d['train']),test_size=len(d['test']),
        unique_spectra=int(len(np.unique(d['groups']))),
        fft_max_error=d['fft_max_error'],linear_baseline=baseline,
        final_metrics={k:float(v) for k,v in metric_rows[-1].items()},
        final_probe_rmse=float(arrays['probe_rmse'][-1]),
        final_sensitivity_relative_error=float(arrays['sensitivity_relative_error'][-1]),
        final_non_dc_weight_power_fraction=float(arrays['weight_power'][-1,:,1:].sum()/
                                                 max(arrays['weight_power'][-1].sum(),1e-25)),
        pca_explained_fraction=pca_summary,
        analysis_notes=[
          'Fourier transforms used for labels/analysis only, never model inputs.',
          'Site index is the cyclic matrix coordinate; output energy rank is not a Fourier coordinate.',
          'Power includes conjugate-pair multiplicity (Parseval normalization).',
          'Fixed PCA fitted jointly across saved checkpoints; these are post-hoc views.',
          'Causal ablations remove real site-mode subspaces before attention blocks and preserve CLS.',
          'Random controls: 3 fixed DC-orthogonal subspaces per rank, matched to removed mode rank.',
          'Ablations use a fixed 128-example held-out subset; loss curves use full train/test sets.',
          'Sensitivity is a central finite difference (epsilon .001) in symmetric coupling coordinates.',
          'Sorted-spectrum sensitivities may average slopes at eigenvalue crossings.',
          'Circulant inputs already carry Fourier structure at initialization; power alone is not causal evidence.',
          'Gini of non-DC power can rise while that power vanishes; inspect absolute power and the non-DC fraction.',
          'These mathematical circulants do not impose carbon valence or graph planarity.' ])
    if target=='mode':
        selectivity=mode_ablation_selectivity(arrays['ablation_output_change'])
        arrays['mode_ablation_selectivity']=selectivity
        summary['mode_ablation_selectivity_initial_by_stage']=selectivity[0].tolist()
        summary['mode_ablation_selectivity_final_by_stage']=selectivity[-1].tolist()
    out=args.run/'analysis'; out.mkdir(exist_ok=True)
    np.savez_compressed(out/'diagnostics.npz',**arrays)
    (out/'summary.json').write_text(json.dumps(jsonable(summary),indent=2)+'\n')
    if not args.no_render:
        from render import render
        render(args.run)


if __name__=='__main__': main()
