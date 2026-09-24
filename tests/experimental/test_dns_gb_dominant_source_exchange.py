"""Cheap Stage-4R algebra and design tests; no real LISA evaluations."""
import numpy as np
import pytest
from examples.lisa_dns_stage4.dominant_source_exchange import (
    source_ranks,dominant_labels,fixed_swap,joint_log_ratio,pair_cases,uniform_labels,correlation)


def test_dominant_tie_breaks_to_lowest_label():
    g=np.array([1,7,7,2,3,4,5,6,0.])
    assert dominant_labels(g)==1
    assert dominant_labels(np.zeros(9))==0


def test_source_rank_bookkeeping_all_labels_and_batches():
    g=np.array([[1,7,7,2,3,4,5,6,0.],np.arange(9)])
    r=source_ranks(g)
    np.testing.assert_array_equal(r[0],[8,1,2,7,6,5,4,3,9])
    np.testing.assert_array_equal(np.sort(r,axis=1),np.tile(np.arange(1,10),(2,1)))
    np.testing.assert_array_equal(dominant_labels(g),[1,8])
    with pytest.raises(ValueError):source_ranks(np.full(9,np.nan))


def test_fixed_swap_moves_both_blocks_exactly_and_preserves_rest():
    x=np.arange(54,dtype=float);y=-x-100
    xx,yy=fixed_swap(x,y,2,7)
    assert xx[12:18].tobytes()==y[42:48].tobytes()
    assert yy[42:48].tobytes()==x[12:18].tobytes()
    assert xx[:12].tobytes()==x[:12].tobytes() and xx[18:].tobytes()==x[18:].tobytes()
    assert yy[:42].tobytes()==y[:42].tobytes() and yy[48:].tobytes()==y[48:].tobytes()


def test_fixed_swap_is_involution_for_every_label_pair():
    x=np.sin(np.arange(54));y=np.cos(np.arange(54))
    for b in range(9):
        for c in range(9):
            xx,yy=fixed_swap(x,y,b,c);xr,yr=fixed_swap(xx,yy,b,c)
            assert xr.tobytes()==x.tobytes() and yr.tobytes()==y.tobytes()


def test_joint_prior_cancels_for_identical_nonlogistic_block_priors():
    # A non-Logistic block density demonstrates independence from Logistic cancellation.
    def prior(z):
        z=z.reshape(9,6)
        return -np.sum(z*z)-np.sum((z[:,0]-z[:,1])**2)
    x=np.linspace(-2,2,54);y=np.sin(np.arange(54))
    for b,c in [(0,8),(3,1),(5,5)]:
        xx,yy=fixed_swap(x,y,b,c)
        assert abs(joint_log_ratio(prior(x),prior(y),prior(xx),prior(yy)))<1e-12


def test_nonidentical_block_priors_need_full_joint_ratio():
    def prior(z):return -np.sum(np.arange(1,10)[:,None]*z.reshape(9,6)**2)
    x=np.ones(54);y=np.zeros(54);xx,yy=fixed_swap(x,y,0,8)
    forward=joint_log_ratio(prior(x),prior(y),prior(xx),prior(yy))
    assert forward==-48
    assert joint_log_ratio(prior(xx),prior(yy),prior(x),prior(y))==-forward


def test_uniform_rng_is_fixed_state_independent_and_no_retries():
    a=uniform_labels();np.testing.assert_array_equal(a,uniform_labels())
    assert a.shape==(448,2) and np.all((a>=0)&(a<9))
    np.testing.assert_array_equal(a,np.random.Generator(np.random.PCG64(20260923)).integers(0,9,(448,2)))
    assert len(np.unique(a))==9


def test_original_bases_are_not_sequentially_updated():
    x=np.zeros(54);y=np.ones(54)
    a,b=fixed_swap(x,y,0,1);c,d=fixed_swap(x,y,2,3)
    assert np.all(c[:6]==0) and np.all(a[:6]==1)
    assert np.all(x==0) and np.all(y==1)
    a[:]=99
    assert np.all(x==0) and np.all(c[:6]==0)


def test_unordered_pair_case_bookkeeping():
    cases=pair_cases();assert len(cases)==448 and len(set(cases))==448
    assert all(i<j for i,j,t in cases)
    for i in range(8):
        for j in range(i+1,8):assert [t for a,b,t in cases if (a,b)==(i,j)]==list(range(16))


def test_constant_rank_has_no_claimed_predictive_correlation():
    assert correlation(np.ones(10),np.arange(10))['rho'] is None
    assert correlation(np.arange(10),np.arange(10))['rho']==pytest.approx(1.)


def test_symmetric_matrix_keeps_zero_cells_and_missing_diagonal():
    from examples.lisa_dns_stage4.dominant_source_exchange import symmetric_matrix
    rows=[dict(i=i,j=j,joint_survives=(i==0 and t<8)) for i,j,t in pair_cases()]
    m=symmetric_matrix(rows)
    np.testing.assert_array_equal(m,m.T)
    assert np.isnan(m.diagonal()).all()
    assert m[0,1]==.5 and m[1,2]==0
    with pytest.raises(AssertionError):symmetric_matrix(rows[:-1])


def test_actual_implemented_prior_source_without_loading_likelihood():
    # Extract only deterministic prior functions, avoiding model/data construction.
    import ast
    import jax
    import jax.numpy as jnp
    from examples.lisa_dns_stage4.prior_refresh_audit import MODEL
    if not MODEL.exists():pytest.skip('External BayesLISAx source unavailable')
    tree=ast.parse(MODEL.read_text())
    names={'u_to_f0_unordered','u_to_box','box_log_jacobian','unpack_theta_u','logprior'}
    factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make_loglike_and_logprior')
    nodes=[n for n in tree.body + factory.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==5
    for n in nodes:n.decorator_list=[]
    eps=np.finfo(np.float64).tiny
    box={n:(0.,1.) for n in ['f0','fdot','iota','psi','lam','beta']}
    ns=dict(jnp=jnp,jax=jax,DTYPE=jnp.float64,EPS=eps,sigmoid=jax.nn.sigmoid,
            Kmax=9,use_gates=False,marg_Aphi=True,order_f0=False,
            f_min_cfg=0.,f_max_cfg=1.,prior_box=box,gaussian_priors={})
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(MODEL),'exec'),ns)
    with jax.enable_x64():
        prior=ns['logprior'];x=np.linspace(-5,5,54);y=np.cos(np.arange(54))*4
        x[0]=36.
        for b,c in [(0,8),(3,1),(5,5)]:
            xx,yy=fixed_swap(x,y,b,c)
            assert abs(joint_log_ratio(prior(x),prior(y),prior(xx),prior(yy)))<1e-10


def test_correlation_accepts_boolean_survival():
    result=correlation([1,2,3,4],[False,False,True,True])
    assert result['rho']==pytest.approx(0.8944271909999159)
