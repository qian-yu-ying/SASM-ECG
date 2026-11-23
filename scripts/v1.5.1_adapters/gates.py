from core import *
from scipy.special import logit

def main():
    pin(10);start=time.time()
    assert sha(RUN/'SASM_v1.5.1_BSPC_Adapter_Server_Spec_LOCKED.md')==SPEC_SHA
    base=read(OUT/'gate_prediction_replay.json');assert base['all_passed']
    episode_errors=[r['abs_diff_pp'] for r in base['checks'] if r['name'].startswith('episode/')]
    assert max(episode_errors)<=1e-9
    displays=[r for r in base['checks'] if r['name'].startswith(('paper/','original_spec/'))]
    assert all(round(r['actual_pp'],2)==r['reference_pp'] for r in displays)
    cal=pd.read_csv(OLD/'calibration_episodes.csv')
    rr=pd.read_csv(DCA/'locked_threshold_episodes.csv');dd=pd.read_csv(DCA/'dca_episodes.csv')
    temp=read(OLD/'temperature.lock.json')['selected'];cut=read(DCA/'global_thresholds.lock.json')['selected']
    errors=[];checks=0;maxerr=max(episode_errors);coverage=[];rows=[];drows=[];rrows=[]
    def equal(label,actual,expected):
        nonlocal checks,maxerr
        delta=abs(float(actual)-float(expected))*100;checks+=1;maxerr=max(maxerr,delta)
        if not np.isfinite(delta) or delta>1e-9:errors.append(dict(label=label,actual=float(actual),expected=float(expected),abs_diff_pp=delta))
    methods=CONTROLS+['Direct-LP-Refit-b']
    refs=[]
    for p in [OLD/'calibration_episodes.csv',OLD/'temperature.lock.json',DCA/'global_thresholds.lock.json',DCA/'locked_threshold_episodes.csv',DCA/'locked_threshold_summary.csv',DCA/'dca_episodes.csv',DCA/'dca_summary.csv']:
        refs.append(dict(path=str(p),sha256=sha(p)))
    assert refs[0]['sha256']=='bf822231010ceb573fc332551b72bc52f67ecd07df526d34fd6e9f5dedc34ddd'
    calidx=cal.set_index(['backbone','cohort','shot','seed','method','calibration'])
    ridx=rr.set_index(['backbone','cohort','shot','seed','method','calibration','rule'])
    didx=dd.set_index(['backbone','cohort','shot','seed','method','calibration','pt'])
    for bb in ['ECGFounder','MERL']:
        for ds in ['chapman','ningbo','cpsc2018','sph']:
            qi,y=H.labels(ds);ns=10 if ds=='chapman' else 20
            for k in [5,10]:
                for s in range(ns):
                    p=OLD/'base/main'/bb/ds/f'shot{k}_seed{s:02d}.npz';z=np.load(p)
                    su=np.load(DCA/'support'/bb/ds/f'shot{k}_seed{s:02d}.npz') if ds!='chapman' else None
                    for m in methods:
                        for mode,T in [('Raw',1.),('DevTemperature',temp[bb][m]['T'])]:
                            scores=z[m]/T;pr=expit(scores)
                            ev=[]
                            for c in range(y.shape[1]):
                                ix=np.minimum((pr[:,c]*10).astype(int),9);ev.append(abs(np.bincount(ix,weights=pr[:,c]-y[:,c],minlength=10)).sum()/len(y))
                            value=float(np.mean(ev));ref=calidx.loc[(bb,ds,k,s,m,mode)]
                            equal(f'calibration/{bb}/{ds}/{k}/{s}/{m}/{mode}/ECE',value,ref.ece)
                            rows.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,calibration=mode,ece=value,ap=float(ref.ap)))
                            if ds=='chapman':continue
                            for pt in PTS:
                                v=dca_values(y,scores,pt);r=didx.loc[(bb,ds,k,s,m,mode,pt)]
                                for met in ['NB','standardized_NB']:equal(f'DCA/{bb}/{ds}/{k}/{s}/{m}/{mode}/{pt}/{met}',v[met],r[met])
                                drows.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,calibration=mode,pt=pt,**v))
                            for rule in ['A_ChapmanGlobal','B_SupportNegative']:
                                threshold=np.full(y.shape[1],cut[bb][m]['raw_logit_threshold']) if rule.startswith('A') else su[m+'__threshold']
                                decision=scores>threshold/T
                                assert np.array_equal(decision,z[m]>threshold)
                                co=confusion(y,decision);r=ridx.loc[(bb,ds,k,s,m,mode,rule)]
                                v={met:float(co[met].mean()) for met in ['specificity','sensitivity','PPV','f1']}
                                for met,a in v.items():equal(f'Threshold/{bb}/{ds}/{k}/{s}/{m}/{mode}/{rule}/{met}',a,r[met])
                                rrows.append(dict(backbone=bb,cohort=ds,shot=k,seed=s,method=m,calibration=mode,rule=rule,**v))
                notice('SANITY_EXTENDED',backbone=bb,cohort=ds,shot=k,max_error_pp=maxerr)
            coverage.append(dict(backbone=bb,cohort=ds,episodes=2*ns,discrimination_and_ECE=True,DCA_and_threshold_archived=ds!='chapman'))
    cf=pd.DataFrame(rows);tf=pd.DataFrame(rrows);df=pd.DataFrame(drows)
    cf.to_csv(OUT/'sanity_calibration_replay.csv',index=False);tf.to_csv(OUT/'sanity_threshold_replay.csv',index=False);df.to_csv(OUT/'sanity_dca_replay.csv',index=False)
    display=[]
    for m,ap,raw,devt,rulev in [('Direct-LP',60.10,4.70,4.52,[96.4,59.8,58.6]),('MeanAnchor',59.84,3.65,3.75,[96.2,66.,54.2])]:
        for mode,ref in [('Raw',raw),('DevTemperature',devt)]:
            val=100*cf[(cf.cohort!='chapman')&(cf.method==m)&(cf.calibration==mode)].ece.mean();assert round(val,2)==ref
            display.append(dict(method=m,metric=mode+'_ECE',actual=val,display=ref,decimals=2))
        val=100*cf[(cf.cohort!='chapman')&(cf.method==m)&(cf.calibration=='Raw')].ap.mean();assert round(val,2)==ap
        display.append(dict(method=m,metric='AP',actual=val,display=ap,decimals=2))
        group=tf[(tf.method==m)&(tf.calibration=='Raw')&(tf.rule=='A_ChapmanGlobal')]
        for met,ref in zip(['specificity','sensitivity','PPV'],rulev):
            val=100*group[met].mean();assert round(val,1)==ref
            display.append(dict(method=m,metric='RuleA_'+met,actual=val,display=ref,decimals=1))
    report=dict(all_passed=not errors,Layer_A=dict(passed=not errors,tolerance_pp=1e-9,max_abs_diff_pp=maxerr,numeric_checks=checks+len(episode_errors),errors=errors[:50]),
      Layer_B=dict(passed=True,rounding='2 decimals for AP/AUC/ECE; 1 decimal for Rule A',printed_panel_checks=len(displays),global_checks=display),
      coverage=coverage,archived_rule_DCA_scope='240 external episodes; 40 Chapman episodes have calibration references but no archived Rule/DCA outcome tables',
      prediction_replay_sha256=sha(OUT/'gate_prediction_replay.json'),spec_sha256=SPEC_SHA,references=refs,adapter_fits_started=0,GPU_used=False,seconds=time.time()-start)
    dump(OUT/'sanity_gate_v1.5.1_check.json',report);assert report['all_passed'],errors[:5]
    notice('SANITY_ALL_PASSED',numeric_checks=report['Layer_A']['numeric_checks'],max_error_pp=maxerr,seconds=time.time()-start)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;dump(OUT/'GATE_ERROR.json',dict(traceback=traceback.format_exc()));raise
