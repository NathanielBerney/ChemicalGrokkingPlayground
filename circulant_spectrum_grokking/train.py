"""Train on raw matrices, saving all states needed for temporal diagnostics."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from experiment import ROOT, DATASET, load_data, build_model, gini, rfft_power


@torch.no_grad()
def evaluate(model, x, y, indices, mean, std, batch_size=512):
    total = torch.zeros((),device=x.device)
    absolute = total.clone()
    per_output = torch.zeros(y.shape[1],device=x.device)
    for batch in indices.split(batch_size):
        error = model(x[batch])*std+mean-y[batch]
        total += error.square().sum()
        absolute += error.abs().sum()
        per_output += error.square().sum(0)
    count = len(indices)*y.shape[1]
    return dict(rmse=float((total/count).sqrt()), mae=float(absolute/count),
                per_output_rmse=(per_output/len(indices)).sqrt().cpu().numpy())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DATASET)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--target',choices=['sorted','mode'],required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps',type=int,default=40000)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--train-fraction',type=float,default=.3)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--heads',type=int,default=4)
    p.add_argument('--mlp',type=int,default=512)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight-decay',type=float,default=1.)
    p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--snapshot-every',type=int,default=500)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if min(args.steps,args.batch_size,args.eval_every,args.snapshot_every)<1:
        raise ValueError('steps and intervals must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision('highest')
    d=load_data(args.dataset,args.seed,args.train_fraction)
    device=torch.device(args.device)
    x=torch.as_tensor(d['x'],device=device)
    y=torch.as_tensor(d[args.target],dtype=torch.float32,device=device)
    train=torch.as_tensor(d['train'],device=device)
    test=torch.as_tensor(d['test'],device=device)
    mean,std=d['mean'],d['std']
    config=vars(args).copy()
    config.update(dataset=str(args.dataset.resolve()),outdir=str(args.outdir.resolve()),
                  n=x.shape[-1],target_mean=mean,target_std=std,
                  train_size=len(train),test_size=len(test),dataset_sha256=d['sha256'],
                  fft_max_error=d['fft_max_error'],split='grouped by spectrum rounded to 7 decimals',
                  device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU')
    model=build_model(config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    rng=torch.Generator(device=device).manual_seed(args.seed+1000)
    out=args.outdir
    out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    start=0
    elapsed_base=0.
    if args.resume:
        ck=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        for key in ('target','seed','dataset_sha256','batch_size','width','heads','mlp','layers',
                    'lr','weight_decay','train_fraction'):
            if ck['config'][key]!=config[key]: raise ValueError(f'Resume mismatch: {key}')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        rng.set_state(ck['batch_rng'].cpu())
        start=ck['step']; elapsed_base=ck['elapsed_seconds']
    elif (out/'config.json').exists():
        raise FileExistsError(f'{out} already has a run; use --resume or a new directory')
    if (out/'complete.json').exists():
        (out/'complete.json').unlink()
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    np.savez(out/'split.npz',train=d['train'],test=d['test'],groups=d['groups'])
    # Fixed probes are saved once and never used for optimization/model selection.
    prng=np.random.default_rng(args.seed+2000)
    probes=prng.choice(d['test'],size=min(128,len(test)),replace=False)
    np.save(out/'probe_indices.npy',probes)
    t0=time.perf_counter()
    def save(step):
        ck=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(),step=step,config=config,batch_rng=rng.get_state(),
                elapsed_seconds=elapsed_base+time.perf_counter()-t0)
        temp=out/'latest.tmp'
        torch.save(ck,temp); temp.replace(out/'latest.pt')
        # Archival checkpoints omit optimizer state to reduce disk use.
        torch.save(dict(model=ck['model'],step=step,config=config),
                   out/'checkpoints'/f'step_{step:06d}.pt')
    fields=['step','elapsed_seconds','train_rmse','test_rmse','train_mae','test_mae',
            'weight_fourier_gini','mean_channel_fourier_gini','weight_norm']
    # On resume discard rows newer than the last durable checkpoint.
    metrics=out/'metrics.csv'
    if args.resume:
        rows=list(csv.DictReader(metrics.open()))
        with metrics.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
            w.writerows([row for row in rows if int(row['step'])<=start])
    with metrics.open('a' if args.resume else 'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        if not args.resume: writer.writeheader()
        for step in range(start,args.steps+1):
            if step>start:
                model.train(); optimizer.zero_grad(set_to_none=True)
                if args.batch_size>=len(train):
                    batch=train
                else:
                    batch=train[torch.randint(len(train),(args.batch_size,),generator=rng,device=device)]
                loss=(model(x[batch])-(y[batch]-mean)/std).square().mean()
                if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at {step}')
                loss.backward(); optimizer.step()
            if (step==0 or step % args.eval_every==0 or step==args.steps) and not (args.resume and step==start):
                model.eval()
                tr=evaluate(model,x,y,train,mean,std)
                te=evaluate(model,x,y,test,mean,std)
                weight=model.row_embed.weight.detach().cpu().numpy()
                power=rfft_power(weight,1)[:,1:]
                row=dict(step=step,elapsed_seconds=elapsed_base+time.perf_counter()-t0,
                         train_rmse=tr['rmse'],test_rmse=te['rmse'],train_mae=tr['mae'],test_mae=te['mae'],
                         weight_fourier_gini=float(gini(power.sum(0))),
                         mean_channel_fourier_gini=float(gini(power).mean()),
                         weight_norm=float(sum(v.detach().square().sum() for v in model.parameters()).sqrt()))
                writer.writerow(row); f.flush(); print(json.dumps(row),flush=True)
            if step in {0,1,10,25,50,100,200} or step%args.snapshot_every==0 or step==args.steps:
                save(step)
    (out/'complete.json').write_text(json.dumps(dict(step=args.steps,
          elapsed_seconds=elapsed_base+time.perf_counter()-t0),indent=2)+'\n')


if __name__=='__main__': main()
