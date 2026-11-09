"""Refresh the comparison plot and a local, browser-readable animation gallery."""
import csv
import html
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT, mode_ablation_selectivity


def main():
    runs=[]
    for path in sorted((ROOT/'runs').glob('*/analysis/summary.json')):
        summary=json.loads(path.read_text())
        config=json.loads((path.parents[1]/'config.json').read_text())
        if config['target']=='mode':
            with np.load(path.parent/'diagnostics.npz') as d:
                selectivity=mode_ablation_selectivity(d['ablation_output_change'])
            summary['mode_ablation_selectivity_initial_by_stage']=selectivity[0].tolist()
            summary['mode_ablation_selectivity_final_by_stage']=selectivity[-1].tolist()
            path.write_text(json.dumps(summary,indent=2)+'\n')
        with (path.parents[1]/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
        metrics={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
        runs.append((summary,config,metrics))
    if not runs: raise ValueError('No analyzed runs')
    out=ROOT/'animations'; out.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    colors=plt.cm.tab10(np.arange(len(runs)))
    for (s,c,m),color in zip(runs,colors):
        ax=axs[0 if c['target']=='sorted' else 1]
        label=f"seed {c['seed']}, wd={c['weight_decay']:g}"
        ax.plot(m['step'],m['test_rmse'],label=label,color=color)
        ax.plot(m['step'],m['train_rmse'],color=color,ls='--',alpha=.7,lw=.8)
    for ax,target in zip(axs,['sorted','mode']):
        matching=[r for r in runs if r[1]['target']==target]
        if not matching: continue
        baseline=matching[0][0]['linear_baseline']['test']['rmse']
        if target=='sorted':
            ax.axhline(baseline,color='black',ls=':',label=f'linear control ≈ {baseline:.3f}')
        else:
            ax.text(.03,.03,f'Linear control RMSE ≈ {baseline:.1e}',transform=ax.transAxes,fontsize=9)
        ax.set(xscale='symlog',yscale='log',xlim=(0,max(r[2]['step'][-1] for r in matching)),
               xlabel='Optimizer step',ylabel='Full-spectrum RMSE (|β|)',
               title='Energy-sorted targets' if target=='sorted' else 'Fourier-mode-ordered targets')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Raw 42 × 42 circulant matrices → eigenvalues\nSolid: held-out test · dashed: train · spectrum-grouped splits')
    fig.savefig(out/'comparison.png',dpi=170); plt.close(fig)
    entries=[]
    table=[]
    for s,c,m in runs:
        name=s['run']; final=s['final_metrics']
        items={k:f'{name}/{k}.gif' for k in ['representations','frequencies','causal_ablations']}
        entries.append(dict(name=name,images=items,
            caption=f"{c['target']} targets · seed {c['seed']} · weight decay {c['weight_decay']:g} · "
                    f"{s['checkpoints']} checkpoints · train/test {s['train_size']}/{s['test_size']}"))
        table.append(f"<tr><td>{html.escape(name)}</td><td>{final['train_rmse']:.5f}</td>"
                     f"<td>{final['test_rmse']:.5f}</td><td>{s['linear_baseline']['test']['rmse']:.3g}</td>"
                     f"<td>{s['final_sensitivity_relative_error']:.3f}</td></tr>")
    payload=json.dumps(entries)
    page='''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Circulant spectrum learning</title>
<style>body{font:16px system-ui;max-width:1250px;margin:35px auto;padding:0 22px;color:#17222e;background:#f7f9fc}
h1{font-size:28px}p{line-height:1.6}select,button{font:inherit;padding:8px;margin:5px;border:1px solid #aebccb;border-radius:6px;background:white}
img{width:100%;height:auto;background:white;border-radius:8px}table{border-collapse:collapse;width:100%;background:white}
td,th{padding:9px;text-align:left;border-bottom:1px solid #dde4ed}.note{color:#526376;font-size:14px}a{color:#1758ad}</style>
<h1>Learning eigenvalue spectra from circulant matrices</h1>
<p>Raw matrix inputs, continuous eigenvalue outputs, and no supplied Fourier features.
The mode-ordered control isolates the linear spectral map. The main task predicts eigenvalues in energy order.
Distinct matrices sharing a spectrum stay in the same split.</p>
<img src="comparison.png" alt="Full train and test error histories for all runs">
<p><label>Run <select id="run"></select></label><label>View <select id="view">
<option value="representations">Embeddings and predicted spectra</option>
<option value="frequencies">Fourier modes, Gini, and coupling sensitivities</option>
<option value="causal_ablations">Causal Fourier-mode ablations</option></select></label>
<button id="restart">Restart animation</button> <a id="download">Open GIF</a></p>
<p id="caption" class="note"></p><img id="movie" alt="Training progression animation">
<p class="note">Each frame is a saved checkpoint. Early frames are more closely spaced in optimizer steps.
PCA coordinates, heatmap scales, and axes are fixed across time. Channel order is chosen from the final checkpoint for display only.</p>
<table><thead><tr><th>Run</th><th>Train RMSE</th><th>Test RMSE</th><th>Linear test RMSE</th><th>Sensitivity relative error</th></tr></thead>
<tbody>TABLE_ROWS</tbody></table>
<p class="note">RMSE uses every eigenvalue in the full split. Sensitivity compares central finite differences on four fixed held-out matrices
with the exact response to continuous changes in symmetric couplings; it is a separate local diagnostic.
Ablations use 128 fixed held-out matrices, remove real Fourier subspaces before attention blocks, and preserve CLS.
Random controls match subspace rank, with three draws per rank. Power concentration alone does not establish a Fourier algorithm;
Gini can increase as non-DC weight power vanishes. These fixed-size mathematical circulants do not enforce carbon valence or planarity.</p>
<p><a href="../README.md">Methods and reproduction instructions</a></p>
<script>const runs=RUN_PAYLOAD;const r=document.querySelector('#run'),v=document.querySelector('#view'),m=document.querySelector('#movie');
runs.forEach((x,i)=>{let o=document.createElement('option');o.value=i;o.textContent=x.name;r.appendChild(o)});
function update(restart=false){const x=runs[+r.value],src=x.images[v.value];m.src=src+(restart?'?restart='+Date.now():'');
document.querySelector('#download').href=src;document.querySelector('#caption').textContent=x.caption;}
r.onchange=()=>update();v.onchange=()=>update();document.querySelector('#restart').onclick=()=>update(true);update();</script></html>'''
    (out/'index.html').write_text(page.replace('TABLE_ROWS','\n'.join(table)).replace('RUN_PAYLOAD',payload))
    lines=['# Circulant spectrum animations','',
           'Open [the interactive gallery](index.html) or any GIF below. All animations use actual saved checkpoints.',
           '', '| Run | Train RMSE | Test RMSE | Linear test RMSE | Animations |',
           '|---|---:|---:|---:|---|']
    for s,c,m in runs:
        name=s['run']; f=s['final_metrics']
        links=' · '.join(f'[{label}]({name}/{stem}.gif)' for stem,label in
                         [('representations','Embeddings'),('frequencies','Frequencies'),('causal_ablations','Ablations')])
        lines.append(f"| {name} | {f['train_rmse']:.5f} | {f['test_rmse']:.5f} | {s['linear_baseline']['test']['rmse']:.3g} | {links} |")
    lines+=['','![Learning curves](comparison.png)','','See [experiment methods](../README.md) for the split, controls, and interpretation limits.','']
    (out/'README.md').write_text('\n'.join(lines))
    (ROOT/'results.json').write_text(json.dumps([s for s,_,_ in runs],indent=2)+'\n')
    print(f'Wrote gallery and comparison for {len(runs)} runs',flush=True)


if __name__=='__main__': main()
