"""Standalone GIFs with fixed axes and shared PCA coordinates across training."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from experiment import ROOT, gini

plt.rcParams.update({'font.size':9,'axes.titlesize':10,'figure.dpi':100,
                     'axes.spines.top':False,'axes.spines.right':False})


def save_animation(path,count,draw,fps=6):
    frames=[]
    for i in range(count):
        fig=draw(i)
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).convert(
            'P',palette=Image.Palette.ADAPTIVE,colors=192))
        if i==count-1: fig.savefig(path.with_suffix('.png'),dpi=150)
        plt.close(fig)
    durations=[round(1000/fps)]*count; durations[-1]=1800
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=durations,
                   loop=0,disposal=2,optimize=False)
    print(f'Saved {path}',flush=True)


def padded_bounds(x):
    lo,hi=float(np.min(x)),float(np.max(x))
    pad=max((hi-lo)*.08,1e-4)
    return lo-pad,hi+pad


def render(run):
    run=Path(run)
    config=json.loads((run/'config.json').read_text())
    summary=json.loads((run/'analysis/summary.json').read_text())
    data=np.load(run/'analysis/diagnostics.npz')
    with (run/'metrics.csv').open() as f: rows=list(csv.DictReader(f))
    metrics={k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
    steps=data['steps']; count=len(steps); n=config['n']; freq=np.arange(1,n//2+1)
    label='energy-sorted' if config['target']=='sorted' else 'Fourier-mode-ordered'
    suffix=f"{label} spectrum · seed {config['seed']} · weight decay {config['weight_decay']:g}"
    out=ROOT/'animations'/run.name; out.mkdir(parents=True,exist_ok=True)
    def title(fig,i,detail):
        fig.suptitle(f'Circulant matrix → {suffix}\n{detail} · step {steps[i]:,}',fontsize=13)
    def loss(ax,i):
        take=metrics['step']<=steps[i]
        for key,color,name in [('train_rmse','#2464d8','train'),('test_rmse','#eb6c21','test')]:
            ax.plot(metrics['step'],metrics[key],color=color,alpha=.16,lw=1)
            ax.plot(metrics['step'][take],metrics[key][take],color=color,label=name,lw=1.3)
        ax.axvline(steps[i],color='black',lw=.8,alpha=.6)
        ax.axhline(summary['linear_baseline']['test']['rmse'],color='#8047a3',ls='--',lw=1,label='linear test')
        ax.set(xscale='symlog',yscale='log',xlim=(0,steps[-1]),xlabel='Optimizer step',ylabel='Full-set RMSE (|β|)',title='Generalization')
        vals=np.r_[metrics['train_rmse'],metrics['test_rmse']]
        # Exact linear control can be far below the neural plot; label separately.
        ax.set_ylim(max(vals.min()*.7,1e-5),vals.max()*1.5)
        ax.legend(loc='best',fontsize=7); ax.grid(alpha=.2)
    def pca(ax,key,i,colors,cmap,title_text):
        values=data[key+'_pca']
        artist=ax.scatter(values[i,:,0],values[i,:,1],c=colors,cmap=cmap,
                          vmin=np.min(colors),vmax=np.max(colors),s=20,alpha=.8)
        ax.set(xlim=padded_bounds(values[:,:,0]),ylim=padded_bounds(values[:,:,1]),
               xlabel='Shared PC 1',ylabel='Shared PC 2',
               title=f'{title_text} ({100*summary["pca_explained_fraction"][key]:.1f}% variance)')
        ax.grid(alpha=.15)
        return artist
    def representations(i):
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        title(fig,i,'Fixed PCA coordinates across all checkpoints')
        a=pca(axs[0,0],'weight',i,np.arange(n),'twilight','Input-column weight embeddings')
        fig.colorbar(a,ax=axs[0,0],label='Cyclic site index')
        a=pca(axs[0,1],'row',i,np.arange(n),'twilight',f'Final row tokens · held-out matrix {data["probe_indices"][0]}')
        fig.colorbar(a,ax=axs[0,1],label='Cyclic site index')
        a=pca(axs[1,0],'cls',i,data['coupling_count'],'viridis','CLS representations · 128 held-out matrices')
        fig.colorbar(a,ax=axs[1,0],label='Number of active offsets')
        ax=axs[1,1]
        indices=np.arange(n)
        for j,color in zip(range(3),['#2464d8','#eb6c21','#2a9e70']):
            ax.plot(indices,data['truth'][j],color=color,alpha=.45,lw=2)
            ax.plot(indices,data['prediction'][i,j],color=color,ls='--',lw=1.2,label=f'Matrix {data["probe_indices"][j]}')
        bounds=padded_bounds(np.r_[data['truth'][:3].ravel(),data['prediction'][:,:3].ravel()])
        ax.set(ylim=bounds,xlabel='Energy rank' if config['target']=='sorted' else 'Fourier mode k',
               ylabel='Eigenvalue (|β|)',title='Held-out spectra: solid truth / dashed prediction')
        ax.legend(fontsize=7); ax.grid(alpha=.2)
        return fig
    weight_power=data['weight_power'][:,:,1:]
    weight_normalized=weight_power/np.maximum(weight_power.sum(-1,keepdims=True),1e-20)
    # Fix channel order by its final dominant frequency for readable progression.
    order=np.argsort(weight_normalized[-1].argmax(-1),kind='stable')
    hp=data['hidden_power'][:,-1,:,1:]
    hnorm=hp/np.maximum(hp.sum(-1,keepdims=True),1e-20)
    horder=np.argsort(hnorm[-1].argmax(-1),kind='stable')
    sensitivity_limit=max(2.,float(np.quantile(abs(data['sensitivity']),.995)))
    def frequencies(i):
        fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        title(fig,i,'Fourier diagnostics use the cyclic site axis; DC excluded from power/Gini')
        loss(axs[0,0],i)
        for ax,values,ordering,text in [(axs[0,1],weight_normalized,order,'Input weights: channel mode selectivity'),
                                         (axs[0,2],hnorm,horder,'Final row activations: channel mode selectivity')]:
            im=ax.imshow(values[i,ordering],aspect='auto',origin='lower',vmin=0,vmax=1,
                         extent=(.5,n//2+.5,-.5,config['width']-.5),cmap='magma')
            ax.set(xlabel='Site frequency k',ylabel='Channel (fixed final ordering)',title=text)
            fig.colorbar(im,ax=ax,label='Fraction of non-DC power')
        ax=axs[1,0]
        non_dc=weight_power.sum((1,2))/np.maximum(data['weight_power'].sum((1,2)),1e-20)
        histories=[(gini(weight_power.sum(1)),'aggregate weight Gini','#2464d8'),
                   (gini(weight_power).mean(1),'mean channel Gini','#eb6c21'),
                   (data['sensitivity_relative_error'],'sensitivity relative error','#2a9e70'),
                   (non_dc,'non-DC weight power fraction','#8047a3')]
        for values,name,color in histories:
            ax.plot(steps,values,color=color,alpha=.15)
            ax.plot(steps[:i+1],values[:i+1],label=name,color=color)
        ax.axvline(steps[i],color='black',lw=.8)
        ax.set(xscale='symlog',xlim=(0,steps[-1]),ylim=(0,max(1.1,float(data['sensitivity_relative_error'].max())*1.05)),
               xlabel='Optimizer step',title='Concentration and functional sensitivity')
        ax.legend(fontsize=7); ax.grid(alpha=.2)
        for ax,values,text in [(axs[1,1],data['true_sensitivity'][0],'Exact coupling sensitivity'),
                               (axs[1,2],data['sensitivity'][i,0],'Learned coupling sensitivity')]:
            im=ax.imshow(values,aspect='auto',origin='lower',cmap='coolwarm',
                         vmin=-sensitivity_limit,vmax=sensitivity_limit,
                         extent=(.5,n//2+.5,-.5,n-.5))
            ax.set(xlabel='Symmetric offset r',ylabel='Energy rank' if config['target']=='sorted' else 'Fourier mode k',title=text)
            fig.colorbar(im,ax=ax,label='Δ eigenvalue / Δ coupling')
        return fig
    change=data['ablation_rmse']-data['probe_rmse'][:,None,None]
    random_change=data['random_ablation_rmse']-data['probe_rmse'][:,None,None,None]
    change_bounds=padded_bounds(np.r_[change.ravel(),random_change.ravel()])
    output_limit=max(float(np.quantile(data['ablation_output_change'],.995)),1e-6)
    def causal(i):
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        title(fig,i,'Causal site-mode removal · fixed 128 held-out matrices')
        loss(axs[0,0],i)
        ax=axs[0,1]
        for j,stage in enumerate(data['stages']):
            color=['#2464d8','#eb6c21','#2a9e70'][j%3]
            ax.plot(freq,change[i,j],marker='.',color=color,label=f'Before block {stage+1}')
            # Rank-two controls match k=1,...,20; Nyquist k=21 is rank one.
            random=random_change[i,j,1]
            ax.axhspan(random.min(),random.max(),color=color,alpha=.12)
            ax.axhline(random.mean(),color=color,ls='--',lw=.9)
            ax.scatter([n//2+.3],[random_change[i,j,0].mean()],color=color,marker='x',s=45)
        ax.axhline(0,color='gray',lw=.7)
        ax.set(xlabel='Removed site frequency k',ylabel='Change in probe RMSE (|β|)',ylim=change_bounds,
               title='Dashed/bands: rank-2 random controls; ×: rank-1')
        ax.legend(fontsize=7); ax.grid(alpha=.2)
        for j,ax in enumerate(axs[1]):
            stage=min(j,len(data['stages'])-1)
            im=ax.imshow(data['ablation_output_change'][i,stage].T,aspect='auto',origin='lower',
                         vmin=0,vmax=output_limit,cmap='magma',extent=(.5,n//2+.5,-.5,n-.5))
            ax.set(xlabel='Removed site frequency k',ylabel='Energy rank' if config['target']=='sorted' else 'Fourier mode k',
                   title=f'Prediction change from ablation before block {stage+1}')
            fig.colorbar(im,ax=ax,label='RMS prediction change (|β|)')
        return fig
    save_animation(out/'representations.gif',count,representations)
    save_animation(out/'frequencies.gif',count,frequencies)
    save_animation(out/'causal_ablations.gif',count,causal)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    render(p.parse_args().run)
