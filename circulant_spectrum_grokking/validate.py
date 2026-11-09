"""Numerical checks for target construction, split isolation, and interventions."""
import numpy as np
import torch
from experiment import load_data, MatrixTransformer, real_mode_basis, rfft_power


def main():
    d=load_data()
    assert not set(d['groups'][d['train']]) & set(d['groups'][d['test']])
    assert len(d['train'])+len(d['test'])==len(d['x'])
    assert d['fft_max_error']<1e-10
    n=d['x'].shape[1]
    rng=np.random.default_rng(4)
    v=rng.normal(size=(10,n))
    np.testing.assert_allclose(rfft_power(v,1).sum(1),(v*v).sum(1),atol=1e-12)
    for k in range(n//2+1):
        q=real_mode_basis(n,k)
        np.testing.assert_allclose(q.T@q,np.eye(q.shape[1]),atol=1e-12)
    design=np.column_stack((d['x'][:,0],np.ones(len(d['x']))))
    weights=np.linalg.lstsq(design[d['train']],d['mode'][d['train']],rcond=None)[0]
    rmse=np.mean((design[d['test']]@weights-d['mode'][d['test']])**2)**.5
    assert rmse<1e-10,rmse
    torch.manual_seed(0)
    model=MatrixTransformer(n=n,width=16,heads=2,mlp=32,layers=2)
    x=torch.tensor(d['x'][:3])
    y,hidden=model(x,return_hidden=True)
    assert y.shape==(3,n) and len(hidden)==3
    q=torch.tensor(real_mode_basis(n,3),dtype=torch.float32)
    _,ablated=model(x,return_hidden=True,intervention=(0,q))
    np.testing.assert_allclose(torch.einsum('nr,bnd->brd',q,ablated[0][:,1:]).detach(),0,atol=1e-6)
    torch.testing.assert_close(hidden[0][:,:1],ablated[0][:,:1])
    y.square().mean().backward()
    assert model.row_embed.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    print(f'PASS: FFT labels, disjoint spectrum groups, Parseval, mode projectors, '
          f'linear solution RMSE={rmse:.3g}, forward/backward, intervention')


if __name__=='__main__': main()
