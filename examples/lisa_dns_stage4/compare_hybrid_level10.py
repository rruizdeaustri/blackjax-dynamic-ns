"""Stage-4P read-only construction comparison and source-block diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from examples.lisa_dns_stage4.hybrid_level10_construction import (
    SOURCE,OUT,retained_bank,verify_prefix)
from examples.lisa_dns_stage4.run_ladder_batch import load_checkpoint,NAMES,check_banks
from examples.lisa_dns_stage4.run_ladder_extension import assess
from examples.lisa_dns_stage4.compare_sequential_refresh import load,acceptance
from examples.lisa_dns_stage4.run_kernel_benchmark import autocorrelation_information as ac
from examples.lisa_dns_stage4.logistic_mh_refresh import safe,quantiles,logq,mh,MATERIAL
from examples.lisa_dns_stage4.prior_refresh_audit import block_indices
from examples.lisa_dns_stage4.hybrid_logistic_refresh import sweep_cost
from examples.lisa_dns_stage4.diagnose_source_permutations import DF


def bank_summary(trace,diagnostics,uncertainty):
    ll=trace['loglikelihood'];f0=trace['position'][:,:,::6]
    ess=np.array([[ac(x)['ess'] for x in walker.T] for walker in f0])
    means=ll.mean(1);fractions=np.asarray(diagnostics['walker_fractions'])
    return dict(compression=diagnostics['ratio'],ess=diagnostics['ess'],
        iid_se=diagnostics['naive_iid_se'],within_walker_se=uncertainty['within_walker_block_se'],
        block_means_se=uncertainty['block_means_se'],between_walker_se=uncertainty['between_walker_se'],
        conservative_se=diagnostics['standard_error'],correlated_interval=diagnostics['interval'],
        walker_fractions=fractions,walker_fraction_range=[fractions.min(),fractions.max()],
        walker_fraction_sd=float(fractions.std(ddof=1)),walker_logL_means=means,
        walker_logL_mean_range=[means.min(),means.max()],walker_logL_mean_sd=float(means.std(ddof=1)),
        f0_ess=ess,f0_ess_median=float(np.median(ess)),f0_ess_minimum=float(ess.min()),
        logL_ess=uncertainty['walker_logL_ess'],
        full_tail_diagnostics=diagnostics,full_uncertainty=uncertainty)


def main():
    report=json.loads((OUT/'report.json').read_text())
    assert report['status'] in ['stopped_failed_level10_gate','frozen_level10']
    preflight=json.loads((OUT/'preflight.json').read_text())
    old=json.loads((SOURCE/'report.json').read_text())['rows'][-1]
    t,m,states,_=load_checkpoint(SOURCE/'checkpoint_level9.json')
    assert (OUT/'checkpoint_level9.json').read_bytes()==(SOURCE/'checkpoint_level9.json').read_bytes()
    verify_prefix([float(x) for x in report['frozen_thresholds']],report['frozen_log_masses'],t,m)
    for path,digest in report['file_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    raw={n:load(OUT/f'level10_{n}_trace.npz') for n in NAMES}
    retained={n:retained_bank(a) for n,a in raw.items()}
    old_raw={n:load(SOURCE/f'level10_{n}_trace.npz') for n in NAMES}
    old_retained={n:retained_bank(a) for n,a in old_raw.items()}
    new=assess(retained['selection']['loglikelihood'],retained['calibration']['loglikelihood'],t[-1],m[-1])
    for key in ['threshold','accepted','estimated_log_mass','selection','calibration','selection_uncertainty','calibration_uncertainty']:
        assert safe(new[key])==report['decision'][key]
    assert new['selection']==json.loads((OUT/'selection_candidate.json').read_text())
    old_recomputed=assess(old_retained['selection']['loglikelihood'],old_retained['calibration']['loglikelihood'],t[-1],m[-1])
    for key in ['threshold','accepted','selection','calibration','selection_uncertainty','calibration_uncertainty']:
        assert safe(old_recomputed[key])==old[key]
    starts={n:load(OUT/f'{n}_initial_states.npz') for n in NAMES};check_banks(starts,t[-1])
    for n in NAMES:
        for key in starts[n]:assert starts[n][key].tobytes()==states[n][key].tobytes()
    assert not ({v.tobytes() for v in raw['selection']['position']} & {v.tobytes() for v in raw['calibration']['position']})
    bounds=json.loads(Path('/tmp/lisa_dns_stage4_prior_smoke/report.json').read_text())['prior_audit'][0]['physical_bounds']['f0']
    def freq(x):return bounds[0]+(bounds[1]-bounds[0])*expit(x)
    result=dict(status=report['status'],candidate_threshold=new['threshold'],frozen_through=10 if new['accepted'] else 9,
        lower_prefix_unchanged=True,exact_historical_starts=True,independent_banks=True,
        sticky_mapping=report['sticky_mapping'],mapping_notice=preflight['sticky_mapping_notice'],
        selection_candidate_saved_before_calibration=True,attempts={},refresh={},blocks={},costs={},
        file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__),OUT/'report.json',OUT/'preflight.json',OUT/'selection_candidate.json',
            *[OUT/f'level10_{n}_trace.npz' for n in NAMES]]})
    for kind,traces,row in [('isotropic',old_retained,old),('hybrid_pair',retained,new)]:
        result['attempts'][kind]=dict(threshold=row['threshold'],accepted=row['accepted'],
            banks={n:bank_summary(traces[n],row[n],row[f'{n}_uncertainty']) for n in NAMES})
    for name,a in raw.items():
        n=len(a['walker']);assert n==3072
        np.testing.assert_array_equal(a['retained'],a['iteration']>=128)
        for lp,ll in [('logprior','loglikelihood'),('post_slice_logprior','post_slice_loglikelihood')]:
            assert np.isfinite(a[lp]).all() and np.isfinite(a[ll]).all() and np.all(a[ll]>t[-1])
        result['refresh'][name]=dict(
            all384=acceptance(a,np.ones(n,dtype=bool)),burn128=acceptance(a,a['iteration']<128),
            retained256=acceptance(a,a['iteration']>=128),
            by_walker=[acceptance(a,a['walker']==w) for w in range(8)],
            negative_infinite_prior=int(np.isneginf(a['proposed_logprior']).sum()),
            nonfinite_draws=int(a['nonfinite_draw'].sum()),nonfinite_draw_coordinates=int(a['nonfinite_draw_coordinates'].sum()),
            nonfinite_proposed_likelihood=int(np.sum(~a['nonfinite_draw']&~np.isfinite(a['proposed_loglikelihood']))),
            material_correction_threshold=MATERIAL,material_corrections=int(np.sum(np.abs(a['log_mh_ratio'])>MATERIAL)),
            correction_distribution=quantiles(a['log_mh_ratio'][~a['nonfinite_draw']]),
            survivor_alpha=quantiles(a['alpha'][a['passes_ell9']]))
        summaries=[]
        for w in range(8):
            oldposition=starts[name]['position'][w]
            accum=[dict(walker=w,label=b,mapped_sticky=(report['sticky_mapping'][name][w]==b),
                slice_changes=0,refresh_attempts=0,refresh_accepts=0,
                slice_latent_path_l2=0.,refresh_latent_path_l2=0.,slice_f0_path_bins=0.,
                refresh_f0_path_bins=0.,first_accepted_refresh=None) for b in range(9)]
            for i in range(384):
                j=w*384+i;post=a['post_slice_position'][j];final=a['position'][j]
                assert np.isfinite(post).all() and np.isfinite(final).all()
                ix=block_indices(a['labels'][j]);keep=np.setdiff1d(np.arange(54),ix)
                assert a['proposed_position'][j,keep].tobytes()==post[keep].tobytes()
                np.testing.assert_array_equal(a['proposed_position'][j,ix],a['draw'][j].ravel())
                qo=logq(post[ix]);qn=logq(a['proposed_position'][j,ix])
                assert qo==a['logq_old'][j] and qn==a['logq_new'][j]
                assert a['old_logprior'][j]==a['post_slice_logprior'][j]
                assert a['old_loglikelihood'][j]==a['post_slice_loglikelihood'][j]
                ratio,alpha=mh(a['post_slice_logprior'][j],a['proposed_logprior'][j],qo,qn,
                    a['proposed_loglikelihood'][j],not a['nonfinite_draw'][j])
                np.testing.assert_equal(ratio,a['log_mh_ratio'][j]);assert alpha==a['alpha'][j]
                assert bool(a['uniform'][j]<alpha)==bool(a['accepted'][j])
                expected=a['proposed_position'][j] if a['accepted'][j] else post
                assert final.tobytes()==expected.tobytes()
                assert a['logprior'][j]==(a['proposed_logprior'][j] if a['accepted'][j] else a['post_slice_logprior'][j])
                assert a['loglikelihood'][j]==(a['proposed_loglikelihood'][j] if a['accepted'][j] else a['post_slice_loglikelihood'][j])
                if a['slice_component'][j]!=0:
                    slabs=a['slice_labels'][j];keep_slice=np.setdiff1d(np.arange(54),block_indices(slabs[slabs>=0]))
                    assert post[keep_slice].tobytes()==oldposition[keep_slice].tobytes()
                for b,s in enumerate(accum):
                    ix=block_indices([b]);attempt=b in a['labels'][j];accepted=attempt and a['accepted'][j]
                    s['slice_changes']+=int(not np.array_equal(post[ix],oldposition[ix]))
                    s['refresh_attempts']+=int(attempt);s['refresh_accepts']+=int(accepted)
                    s['slice_latent_path_l2']+=float(np.linalg.norm(post[ix]-oldposition[ix]))
                    s['refresh_latent_path_l2']+=float(np.linalg.norm(final[ix]-post[ix]))
                    s['slice_f0_path_bins']+=float(abs(freq(post[ix[0]])-freq(oldposition[ix[0]]))/DF)
                    s['refresh_f0_path_bins']+=float(abs(freq(final[ix[0]])-freq(post[ix[0]]))/DF)
                    if accepted and s['first_accepted_refresh'] is None:s['first_accepted_refresh']=i+1
                c=sweep_cost(int(a['num_steps'][j]),int(a['num_shrink'][j]),not a['nonfinite_draw'][j])
                for k,v in c.items():assert a[k][j]==v
                oldposition=final
            for b,s in enumerate(accum):
                ix=block_indices([b]);initial=starts[name]['position'][w]
                s.update(latent_net_l2=float(np.linalg.norm(oldposition[ix]-initial[ix])),
                    initial_f0_hz=float(freq(initial[ix[0]])),final_f0_hz=float(freq(oldposition[ix[0]])),
                    f0_net_bins=float((freq(oldposition[ix[0]])-freq(initial[ix[0]]))/DF))
            summaries.extend(accum)
        result['blocks'][name]=summaries
        old_slice=int((old_raw[name]['num_steps']+old_raw[name]['num_shrink']).sum())
        assert old_slice==old['cost'][f'{name}_likelihood_cost_proxy']
        newcost=report['banks'][name]['cost']
        for k in ['slice_likelihood_proxy','slice_prior_proxy','refresh_likelihood','refresh_prior','cache_likelihood','cache_prior']:
            assert newcost[k]==int(a[k].sum())
        result['costs'][name]=dict(isotropic=dict(slice_likelihood_proxy=old_slice,slice_prior_proxy=old_slice,
            refresh_likelihood=0,refresh_prior=0,cache_checks_per_scalar=3080,total_proxy_per_scalar=old_slice+3080),
            hybrid_pair=newcost,hybrid_loop_seconds=sum(report['banks'][name]['walker_seconds']))
    mapped=[s for s in result['blocks']['calibration'] if s['mapped_sticky']]
    result['mapped_sticky_replaced_walkers']=[s['walker'] for s in mapped if s['refresh_accepts']>0]
    result['saved_trace_integrity_passed']=True
    if new['accepted']:
        ft,fm,_,_=load_checkpoint(OUT/'checkpoint_level10.json',(t,m));assert len(ft)==11
        assert ft[-1]==new['threshold'] and fm[-1]==m[-1]+np.log(new['calibration']['ratio'])
        result['frozen_ell10']=float(ft[-1]);result['frozen_logX10']=float(fm[-1])
    else:
        assert not (OUT/'checkpoint_level10.json').exists()
        assert len(report['frozen_thresholds'])==10
    assert not (OUT/'checkpoint_level11.json').exists()
    (OUT/'comparison.json').write_text(json.dumps(safe(result),indent=2,allow_nan=False))
    print('Read-only comparison complete. Gate:',new['accepted'],'calibration ESS:',new['calibration']['ess'])
    print('Mapped sticky blocks replaced in calibration walkers:',result['mapped_sticky_replaced_walkers'])


if __name__=='__main__':main()
