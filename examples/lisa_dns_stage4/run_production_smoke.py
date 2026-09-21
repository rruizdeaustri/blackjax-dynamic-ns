"""One frozen seed44 DNS smoke; consumes IID levels without building any."""
import argparse
import hashlib
import json
from pathlib import Path

from examples.lisa_dns_stage4 import audit
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.ns import dns
from examples.lisa_dns_stage4 import proposals
from examples.lisa_dns_stage4.run_smoke import json_safe


def frozen_inputs(calibration):
    if not calibration['IID_level2']['accepted']:
        raise ValueError('Requires accepted independent IID calibration')
    levels=dns.create_levels([-np.inf,calibration['ell1'],calibration['IID_level2']['selection']['threshold']],
                             [0.,np.log(calibration['Xhat1']),calibration['IID_level2']['log_X2']],np.zeros(3))
    scales=np.full(54,np.pi/np.sqrt(3));scales.setflags(write=False)
    return levels,scales


def highest_eligible(levels,logL):
    eligible=np.asarray(dns.is_eligible(levels,jnp.asarray(logL),jnp.arange(len(levels.log_mass))))
    return int(np.flatnonzero(eligible)[-1])


def complete_excursions(indices):
    """Indices include initialization at zero; return disjoint top-bottom-top trips."""
    start=bottom=None;trips=[]
    for i,level in enumerate(indices):
        if level==2:
            if start is not None and bottom is not None:
                trips.append(dict(start=start,first_bottom=bottom,end=i))
            start,bottom=i,None
        elif level==0 and start is not None and bottom is None:
            bottom=i
    return trips


def level_bookkeeping(assigned,highest,proposed,eligible,accepted):
    assigned=np.asarray(assigned);highest=np.asarray(highest)
    proposed=np.asarray(proposed);eligible=np.asarray(eligible,dtype=bool);accepted=np.asarray(accepted,dtype=bool)
    if len(assigned)!=len(proposed)+1 or len(highest)!=len(assigned):
        raise ValueError('Assigned/highest histories include initialization; proposals do not')
    previous=assigned[:-1];direction=proposed-previous
    matrix=np.zeros((3,3),dtype=int)
    np.add.at(matrix,(previous,assigned[1:]),1)
    directions={}
    for name,sign in [('up',1),('down',-1)]:
        mask=direction==sign
        directions[name]=dict(proposals=int(mask.sum()),eligible=int((mask&eligible).sum()),
                              accepted=int((mask&accepted).sum()))
    return dict(assigned_occupancy=np.bincount(assigned[1:],minlength=3).tolist(),
                highest_eligible_occupancy=np.bincount(highest[1:],minlength=3).tolist(),
                transition_matrix=matrix.tolist(),directions=directions,
                min_assigned=int(assigned.min()),max_assigned=int(assigned.max()),
                complete_excursions=complete_excursions(assigned),
                low_assigned_while_highest_2=int(np.sum((assigned[1:]<2)&(highest[1:]==2))),
                level0_while_highest_2=int(np.sum((assigned[1:]==0)&(highest[1:]==2))),
                boundary_proposals=int(np.sum((proposed<0)|(proposed>2))),
                occupancy_notice='Counts exclude initialization; rows/columns of transition matrix are previous/new assigned levels, including rejected self-loops.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',type=Path,default=Path('/tmp/lisa_dns_stage4_rejection/report.json'))
    parser.add_argument('--initial-families',type=Path,default=Path('/tmp/lisa_dns_stage4_audit/initial_families.npz'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/lisa_dns_stage4_production_smoke'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    calibrated=json.loads(args.calibration.read_text())
    report=dict(notice=audit.NOTICE,status='initializing',seed=14404,family=44,sweeps=512,
                num_inner_steps=1,num_level_steps=1,mixture=[.2,.4,.4],
                initialization_convention='dns.init default assigned level 0; highest eligibility recorded independently',
                level_weights='log_weight=[0,0,0], equal unnormalized weights (1,1,1), normalized (1/3,1/3,1/3)',
                failures=dict(contour=0,cache=0,nonfinite=0,parameter_contract=0,slice=0,catalogue=0))
    records=[];states=[]
    def save():
        (args.output/'report.json').write_text(json.dumps(json_safe(report),indent=2,allow_nan=False))
        if records:
            np.savez_compressed(args.output/'trace.npz',**{k:np.asarray([r[k] for r in records]) for k in records[0]})
        if states:
            np.savez_compressed(args.output/'states.npz',position=[np.asarray(s.particle.position) for s in states],
                loglikelihood=[float(s.particle.loglikelihood) for s in states],assigned_level=[int(s.level_index) for s in states])
    try:
        problem,cfg,config_path=audit.load_problem()
        assert cfg==calibrated['config']
        model_path=audit.ROOT/'BayesLISAx/src/jax_samplers/problems/lisa_gb_transdim_problem.py'
        for p in [model_path,config_path]:
            assert hashlib.sha256(p.read_bytes()).hexdigest()==calibrated['file_sha256'][str(p)]
        prior,like=map(jax.jit,audit.scalar_functions(problem))
        levels,scales=frozen_inputs(calibrated)
        frozen=jax.tree.map(lambda x:np.asarray(x).copy(),levels);original_scales=scales.copy()
        archive=np.load(args.initial_families);match=np.flatnonzero(archive['seeds']==44)
        assert len(match)==1
        initial=jnp.asarray(archive['positions'][match[0]])
        assert initial.shape==(54,) and initial.dtype==jnp.float64
        assert np.isfinite(float(prior(initial)))
        np.testing.assert_allclose(like(initial),-90964.02933083786,rtol=0,atol=1e-6)
        state=dns.init(initial,prior,like,levels)  # Preserve the API's default level zero.
        states.append(state)
        report.update(config=cfg,devices=str(jax.devices()),jax_version=jax.__version__,
            thresholds=np.asarray(levels.loglikelihood),log_masses=np.asarray(levels.log_mass),
            log_weights=np.asarray(levels.log_weight),isotropic_scale=float(scales[0]),
            initial_logprior=float(state.particle.logdensity),initial_logL=float(state.particle.loglikelihood),
            initial_assigned=int(state.level_index),initial_highest_eligible=highest_eligible(levels,state.particle.loglikelihood),
            file_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [args.calibration,args.initial_families,config_path,model_path,Path(dns.__file__),Path(proposals.__file__),Path(__file__)]})
        np.savez(args.output/'frozen_inputs.npz',thresholds=levels.loglikelihood,log_mass=levels.log_mass,
                 log_weight=levels.log_weight,scales=scales,initial=initial)
        parameter=proposals.build_parameter_step(prior,like,scales)
        kernel=jax.jit(dns.build_kernel(levels,parameter,num_inner_steps=1,num_level_steps=1))
        key=jax.random.key(report['seed'])
        save()
        for iteration in range(1,513):
            previous=int(state.level_index)
            key,subkey=jax.random.split(key);state,info=kernel(subkey,state)
            p=state.particle;index=int(state.level_index)
            pi=jax.tree.map(lambda a:a[0],info.parameter_info)
            li=jax.tree.map(lambda a:a[0],info.level_info)
            if not np.all(np.asarray(info.parameter_is_valid)):
                report['failures']['parameter_contract']+=1;raise ValueError('Invalid parameter callback')
            if not (np.all(np.isfinite(p.position)) and np.isfinite(float(p.logdensity)) and np.isfinite(float(p.loglikelihood))):
                report['failures']['nonfinite']+=1;raise ValueError('Nonfinite state')
            if not bool(dns.is_eligible(levels,p.loglikelihood,index)) or not bool(dns.is_eligible(levels,p.loglikelihood,previous)):
                report['failures']['contour']+=1;raise ValueError('Contour failure')
            try:
                np.testing.assert_allclose(p.logdensity,prior(p.position),rtol=0,atol=1e-9)
                np.testing.assert_allclose(p.loglikelihood,like(p.position),rtol=0,atol=1e-7)
            except AssertionError:
                report['failures']['cache']+=1;raise
            report['failures']['slice']+=int(not bool(pi.slice.is_accepted))
            if not bool(pi.slice.is_accepted):
                raise ValueError('Slice failure; stopping for diagnosis')
            records.append(dict(iteration=iteration,position=np.asarray(p.position),logprior=float(p.logdensity),
                loglikelihood=float(p.loglikelihood),assigned_level=index,highest_eligible_level=highest_eligible(levels,p.loglikelihood),
                previous_level=int(li.previous_level),proposed_level=int(li.proposed_level),
                level_proposal_direction=int(li.proposed_level-li.previous_level),level_eligible=bool(li.is_eligible),
                level_accepted=bool(li.is_accepted),level_acceptance_probability=float(li.acceptance_probability),
                component=int(pi.component),labels=np.asarray(pi.labels),num_steps=int(pi.slice.num_steps),
                num_shrink=int(pi.slice.num_shrink),slice_accepted=bool(pi.slice.is_accepted),
                cost_proxy=int(pi.slice.num_steps+pi.slice.num_shrink)))
            states.append(state)
            if iteration%128==0:
                print('production',iteration,'assigned',index,'highest',records[-1]['highest_eligible_level'],'logL',float(p.loglikelihood),flush=True)
        jax.tree.map(np.testing.assert_array_equal,levels,frozen)
        np.testing.assert_array_equal(scales,original_scales)
        assigned=[int(s.level_index) for s in states]
        highest=[report['initial_highest_eligible']]+[r['highest_eligible_level'] for r in records]
        summary=level_bookkeeping(assigned,highest,[r['proposed_level'] for r in records],
                                 [r['level_eligible'] for r in records],[r['level_accepted'] for r in records])
        report['levels_summary']=summary
        values=np.array([r['loglikelihood'] for r in records])
        report['production']=dict(length=len(records),min_logL=float(values.min()),max_logL=float(values.max()),
            best_logL=float(values.max()),best_including_initial=float(max(values.max(),report['initial_logL'])),
            final_logL=float(values[-1]),likelihood_cost_proxy=sum(r['cost_proxy'] for r in records),
            direct_cache_likelihood_evaluations=len(records),
            initial_likelihood_evaluations=2,latent_displacement=float(np.linalg.norm(np.asarray(states[-1].particle.position-initial))),
            component_counts=np.bincount([r['component'] for r in records],minlength=3).tolist())
        # Checkpoints refer to iteration zero (initialization) or completed sweep numbers.
        checkpoints={'initial_seed44':0,'best_production':int(np.argmax(values))+1,'final':512}
        low=next((i for i in range(1,len(states)) if assigned[i]==0),None)
        if low is not None: checkpoints['first_visit_level0']=low
        returned=next((i for i in range((low or 0)+1,len(states)) if assigned[i]==2),None) if low is not None else None
        if returned is not None: checkpoints['first_return_level2_after_level0']=returned
        trips=summary['complete_excursions']
        if trips:
            trip=trips[0]
            checkpoints.update(witness_start=trip['start'],witness_bottom=trip['first_bottom'],witness_end=trip['end'])
        decoded={}
        for i in sorted(set(checkpoints.values())):
            cat=audit.catalogue(problem,states[i].particle.position)
            try:
                np.testing.assert_allclose(cat['loglikelihood'],states[i].particle.loglikelihood,rtol=0,atol=1e-7)
                for k in ['residual_chi2','logdet','source_gain','amplitude','phase']:
                    assert np.all(np.isfinite(cat[k]))
                assert cat['residual_chi2']>=0
                f0=np.asarray(cat['decoded']['f0']).reshape(9)
                assert np.all(np.isfinite(f0))
                assert np.all((f0>=problem.f_min_cfg)&(f0<=problem.f_max_cfg))
            except AssertionError:
                report['failures']['catalogue']+=1;raise
            cat.update(iteration=i,assigned_level=assigned[i],highest_eligible_level=highest[i],f0=f0.tolist(),
                latent_displacement=float(np.linalg.norm(np.asarray(states[i].particle.position-initial))))
            decoded[i]=cat
        initial_f0=np.array(decoded[0]['f0'])
        for cat in decoded.values(): cat['f0_displacement_from_initial']=(np.array(cat['f0'])-initial_f0).tolist()
        report['checkpoints']={name:decoded[i] for name,i in checkpoints.items()}
        report['sparse_reconstruction_count']=len(decoded)
        report['production']['physical_f0_displacement']=report['checkpoints']['final']['f0_displacement_from_initial']
        np.savez(args.output/'checkpoints.npz',names=list(checkpoints),iterations=list(checkpoints.values()),
                 position=[np.asarray(states[i].particle.position) for i in checkpoints.values()])
        if trips:
            a,b=trip['start'],trip['end'];before,after=decoded[a],decoded[b]
            fdelta=np.array(after['f0'])-np.array(before['f0'])
            block_delta=np.linalg.norm(np.asarray(states[b].particle.position-states[a].particle.position).reshape(9,6),axis=1)
            accepted_path=[dict(iteration=i,previous=assigned[i-1],assigned=assigned[i],logL=float(states[i].particle.loglikelihood))
                           for i in range(a+1,b+1) if assigned[i]!=assigned[i-1]]
            report['mechanism_witness']=dict(**trip,start_logL=before['loglikelihood'],end_logL=after['loglikelihood'],
                minimum_assigned=min(assigned[a:b+1]),accepted_transition_path=accepted_path,
                changed_source_labels=np.flatnonzero(block_delta>1e-8).tolist(),
                frequency_changed_labels=np.flatnonzero(np.abs(fdelta)>1e-12).tolist(),
                change_definition='label latent-block norm >1e-8; frequency change >1e-12 Hz; no family classification',
                f0_before=before['f0'],f0_after=after['f0'],f0_displacement=fdelta.tolist(),
                residual_chi2_before=before['residual_chi2'],residual_chi2_after=after['residual_chi2'])
            np.savez(args.output/'mechanism_witness.npz',iterations=np.arange(a,b+1),
                position=[np.asarray(states[i].particle.position) for i in range(a,b+1)],assigned_level=assigned[a:b+1],
                highest_eligible_level=highest[a:b+1],loglikelihood=[float(states[i].particle.loglikelihood) for i in range(a,b+1)])
        report['success']=bool(not any(report['failures'].values()) and len(set(assigned))>1 and
            summary['directions']['up']['accepted']>0 and summary['directions']['down']['accepted']>0 and trips and
            report['production']['latent_displacement']>1e-6)
        report['status']='completed_success' if report['success'] else 'completed_needs_diagnosis'
    except Exception as exc:
        report.update(status='stopped_error',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save();print('Report:',args.output/'report.json',report['status'],flush=True)


if __name__=='__main__':main()
