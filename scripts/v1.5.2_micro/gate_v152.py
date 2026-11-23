from solver import *
from concurrent.futures import ProcessPoolExecutor,as_completed

def check(job):
    bb,ds,k,s=job;pin();ep=episode(bb,ds,k,s);w,b,d=solve(ep,.1,10.,'zero_unmatched');pred=ep['xq']@w+b
    old=old_prediction(bb,ds,k,s);qi,y=H.labels(ds);assert np.array_equal(qi,ep['qi']) and np.array_equal(qi,old['query_indices'])
    err=float(abs(pred-old['MeanAnchor']).max());aperr=abs(macro_ap(y,pred)-macro_ap(y,old['MeanAnchor']))*100
    return dict(backbone=bb,cohort=ds,shot=k,seed=s,max_abs_logit_diff=err,macro_ap_error_pp=aperr,passed=err<=1e-6 and aperr<=1e-9,solver=d)

def main():
    OUT.mkdir(parents=True,exist_ok=True);pin(10);assert sha(RUN/SPEC)==SPEC_SHA
    assert sha(V151/'SASM_v1.5.1_BSPC_Adapter_Server_Spec_LOCKED.md')==SPEC151_SHA
    assert read(V151/'results/v1.5.1_adapters/sanity_gate_v1.5.1_check.json')['all_passed']
    before=history_snapshot();dump(RUN/'HISTORICAL_TREE_BEFORE.json',before)
    for n in ['MAIN_FITS_COMPLETE.json','MAIN_SCORING_COMPLETE.json','SWAP_FITS_COMPLETE.json','SWAP_SCORING_COMPLETE.json','BOOTSTRAPS_COMPLETE.json','TIMING_COMPLETE.json','BUNDLE_COMPLETE.json']:assert (V151/n).exists() and read(V151/n)
    draw=read(V151/'results/v1.5.1_adapters/bootstrap_draws_manifest.json');refs=[]
    for row in draw['provenance']:
        if row['subgroup'] is None:
            p=V151/row['file'];actual=sha(p);assert actual==row['sha256'];refs.append(dict(path=str(p),sha256=actual))
    cfg=read(V151/'results/v1.5.1_adapters/dev_chapman_grid_search.json')
    for bb in BB:
        c=cfg['selected'][bb]['TaskRes-ECG/BCE']['config'];assert abs(c['lambda_res']/c['gamma']**2-40)<1e-12
    jobs=[(bb,'chapman',k,s) for bb in BB for k in SHOTS for s in range(10)]+[(bb,ds,k,0) for bb in BB for ds in DS for k in SHOTS]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(check,j) for j in jobs]):
            r=f.result();rows.append(r);notice('GATE_REPLAY',done=len(rows),total=len(jobs),max_logit_error=r['max_abs_logit_diff'],ap_error_pp=r['macro_ap_error_pp'])
    passed=all(v['passed'] for v in rows);assert history_snapshot()==before
    dump(OUT/'gate_v152_check.json',dict(all_passed=passed,spec_sha256=SPEC_SHA,v151_spec_sha256=SPEC151_SHA,history_unchanged=True,draw_files=refs,episodes=rows,max_abs_logit_diff=max(v['max_abs_logit_diff'] for v in rows),max_macro_ap_error_pp=max(v['macro_ap_error_pp'] for v in rows)))
    if not passed:raise RuntimeError('Stage 0 failed: no later module is authorized to run')
    notice('STAGE0_PASSED',episodes=len(rows))

if __name__=='__main__':
    try:main()
    except Exception:
        p=OUT/'gate_v152_check.json';v=read(p) if p.exists() else dict(spec_sha256=SPEC_SHA)
        v.update(all_passed=False,error=traceback.format_exc());dump(p,v);raise
