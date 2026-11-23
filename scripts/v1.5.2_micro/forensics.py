"""Read-only cache/source evidence. Never executes an encoder or extraction script."""
from micro_common import *
import subprocess,re

def stats(v):
    v=np.asarray(v);return dict(count=int(v.size),signed_median=float(np.median(v)) if v.size else None,median_abs=float(np.median(abs(v))) if v.size else None,p95_abs=float(np.quantile(abs(v),.95)) if v.size else None,max_abs=float(abs(v).max()) if v.size else None,undefined_reason=None if v.size else 'no records in this stratum')

def main():
    pin(10);guard();assert (RUN/'MODULE3_COMPLETE.json').exists();R=ROOT.parent
    sources=[R/'extract_features_1024.py',R/'extract_features_filtered.py',R/'extract_cpsc2018_features_timecorrected.py',ROOT/'cpsc9_rerun_20261117/extract_features.py',H.B/'source/locked_ft_runner.py',OLD/'common.py',H.B/'source/target_contract.py']
    evidence=[];clips=[]
    # Read source text only. Window extraction modules have CUDA assertions and
    # must never be imported or run in this CPU forensic stage.
    for p in sources:
        if not p.exists():continue
        lines=p.read_text(errors='replace').splitlines();selected=[]
        for i,line in enumerate(lines):
            if any(k in line for k in ['sigmoid','probs_sum','feat_sum','logits_sum','n_windows','win_count','mean_pool','feat[pos','window','clip(']):
                selected.append(dict(line=i+1,text=line[:300]))
        evidence.append(dict(path=str(p),sha256=sha(p),source_lines=selected))
    # Limit provenance inspection to named consumers in this 12-lead study;
    # never collect source snippets from unrelated project branches.
    consumers=['cs_sasm_experiment.py','corrected_external_validation.py','chapman_validation.py','sasm_dev_matrix.py','ptbxl_matched_b3_extension.py','ptbxl_shared_episode_benchmark.py']
    for p in [R/n for n in consumers if (R/n).exists()]:
        text=p.read_text(errors='replace')
        if not any(k in text.lower() for k in ['ptb','chapman','cpsc','ningbo','sph']):continue
        for i,line in enumerate(text.splitlines()):
            if 'clip' in line and any(k in line for k in ['prob','1e-6','1e-7','EPS']):clips.append(dict(path=str(p),sha256=sha(p),line=i+1,text=line.strip()[:300],association='candidate clipping evidence; not proof this function generated every source head'))
    dense=np.load(H.B/'assets/dense.npz');P=dense['Dw'];q=dense['Db'];assert P.shape==(150,1024) and q.shape==(150,)
    cohorts={}
    for ds in ['chapman','ningbo','cpsc2018','sph']:
        mp=H.O/'manifest.csv' if ds=='cpsc2018' else H.B/'data'/ds/'manifest.csv';md=pd.read_csv(mp);nw=md.windows.to_numpy(int)
        fp=H.O/'ECGFounder_features.npy' if ds=='cpsc2018' else H.B/'data'/ds/'features.npy'
        cohorts[ds]=dict(core_manifest=str(mp),core_manifest_sha256=sha(mp),core_h_cache=str(fp),records=len(md),T1_records=int((nw==1).sum()),Tgt1_records=int((nw>1).sum()),Tgt1_fraction=float((nw>1).mean()),window_median=float(np.median(nw)),window_p95=float(np.quantile(nw,.95)),window_max=int(nw.max()),T_distribution={str(v):int((nw==v).sum()) for v in np.unique(nw)},paired_core_g_status='No one-to-one exported probability cache has yet been proven to match these exact core h vectors; candidate historical caches audited separately below.')
    dbpath=R/'data/PTBXL_v1.0.3/ptbxl_database.csv';db=pd.read_csv(dbpath);train=db[db.strat_fold<=8]
    cohorts['ptbxl_folds1_8']=dict(database=str(dbpath),database_sha256=sha(dbpath),records=len(train),T1_records=len(train),Tgt1_records=0,Tgt1_fraction=0.,window_median=1.,window_p95=1.,window_max=1,evidence='PTB-XL filename_hr records are fixed 10-second records; paired-cache n_windows verified separately. This count is folds 1-8 before the source-head label/record filtering.')
    candidates=[('ptbxl',R/'res/features_1024/ptbxl_feat1024_filtered.npz'),('chapman',R/'res/features_1024/chapman_feat1024_filtered.npz'),('ningbo',R/'res/features_1024/ningbo_feat1024_filtered.npz'),('chapman',R/'res/features_1024/chapman_feat1024.npz'),('ningbo',R/'res/features_1024/ningbo_feat1024.npz'),('cpsc2018',R/'res/features_1024/cpsc2018_feat1024_timecorrected.npz')]
    pairs=[]
    # A candidate clipping constant is evaluated explicitly, without asserting
    # undocumented historical provenance. Preserve the unclipped saturation
    # information and the available mean logits as independent evidence.
    for ds,p in candidates:
        if not p.exists():pairs.append(dict(cohort=ds,path=str(p),available=False,missing_reason='candidate cache absent'));continue
        a=np.load(p);keys=a.files
        if not all(k in keys for k in ['feat','probs']):pairs.append(dict(cohort=ds,path=str(p),available=True,keys=keys,missing_reason='paired feat/probs fields absent'));continue
        h=a['feat'].astype(float);prob=a['probs'].astype(float);nw=a['n_windows'] if 'n_windows' in keys else np.ones(len(h),dtype=int);valid=a['valid'].astype(bool) if 'valid' in keys else np.ones(len(h),bool)
        window_source='cache n_windows' if 'n_windows' in keys else 'single-pass extractor code; current Chapman/Ningbo manifests independently have T=1 for every record'
        if ds=='ptbxl':
            rec=a['records'].astype(str);ids=[]
            for s in rec:
                m=re.match(r'(\d+)',Path(s).stem);ids.append(int(m.group(1)) if m else -1)
            valid&=np.isin(ids,train.ecg_id)
        linear=h@P.T+q;strata={}
        for name,mask in [('T1',valid&(nw==1)),('Tgt1',valid&(nw>1))]:
            evaluations={}
            for eps in [1e-6,1e-7]:
                adjusted=logit(np.clip(prob[mask],eps,1-eps));evaluations[str(eps)]=dict(delta_tilde_g_minus_linear=stats(adjusted-linear[mask]),clipped_fraction=float(((prob[mask]<=eps)|(prob[mask]>=1-eps)).mean()) if mask.any() else None)
            row=dict(records=int(mask.sum()),epsilon_sensitivity=evaluations,unclipped_saturated_fraction=float(((prob[mask]<=0)|(prob[mask]>=1)).mean()) if mask.any() else None)
            if 'logits_mean' in keys:row['mean_logit_minus_Pmeanh_q']=stats(a['logits_mean'][mask].astype(float)-linear[mask])
            strata[name]=row
        match=dict(status='not_checked_no_core_ptb_h_cache')
        if ds in cohorts:
            md=pd.read_csv(cohorts[ds]['core_manifest']);norm=lambda s:str(s).replace(chr(92),'/')
            records=[norm(s) for s in a['records'].astype(str)];assert len(set(records))==len(records)
            lookup={s:i for i,s in enumerate(records)};pairs_ix=[(i,lookup[norm(r)]) for i,r in enumerate(md.record_id) if norm(r) in lookup]
            if pairs_ix:
                f=np.load(cohorts[ds]['core_h_cache'],mmap_mode='r');err=0.
                for st in range(0,len(pairs_ix),1024):
                    block=pairs_ix[st:st+1024];ii,jj=zip(*block);err=max(err,float(abs(f[list(ii)].astype(float)-h[list(jj)]).max()))
                match=dict(status='matches_core_h_within_1e-5' if err<=1e-5 else 'record_matched_but_not_exact_core_h',matched_records=len(pairs_ix),core_records=len(md),max_abs_h_error=err,comparison_complete_for_matching_records=True,interpretation='Do not infer identical preprocessing or an exact core probability interface from record IDs alone.')
            else:match=dict(status='record_ID_alignment_unavailable',sample_records=0)
        pairs.append(dict(cohort=ds,path=str(p),sha256=sha(p),available=True,keys=keys,records=int(valid.sum()),branch='historical_filtered' if 'filtered' in p.name else 'historical_timecorrected' if ds=='cpsc2018' else 'historical_singlepass_unfiltered',window_count_source=window_source,provenance_caveat='These paired-cache errors are not asserted to describe the current core pipeline unless exact record/feature alignment is established.',core_h_alignment=match,strata=strata))
    try:commit=subprocess.run(['git','-C',str(R),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip() or None
    except OSError:commit=None
    out=dict(all_read_only=True,no_encoder_forward=True,no_GPU=True,source_commit=commit,source_evidence=evidence,dense_head=dict(path=str(H.B/'assets/dense.npz'),sha256=sha(H.B/'assets/dense.npz'),P_shape=list(P.shape),q_shape=list(q.shape)),cohorts=cohorts,paired_cache_audits=pairs,
      aggregation_findings=dict(historical_filtered='Source sums per-window hidden features, sigmoid probabilities and logits separately; each divided by the window count. Thus logit(mean(sigmoid(logit))) need not equal mean(logit).',new_CPSC_core='20261117 extraction saves averaged window hidden features; no corresponding 150-class probabilities are written by that script.',core_h_vs_candidate_g='Do not silently equate historical filtered probability exports with the current core h caches.'),
      clipping_provenance=dict(status='not_uniquely_resolved_for_all_core_source_heads',candidate_source_lines=clips,evaluated_epsilons=[1e-6,1e-7],note='EPS in waveform z-scoring is not evidence for probability-logit clipping. No epsilon is invented as the proven historical value.'),
      missing_items=['A probability export proven to match every current core h cache was not located; reported historical branch errors are labeled separately.','SPH paired historical probability cache was not located in the scoped feature-cache directory.','Per-window hidden/logit arrays were not re-extracted; no causal decomposition beyond the available cache/code evidence is claimed.'])
    dump(OUT/'ecgfounder_interface_forensics.json',out);dump(RUN/'MODULE4_COMPLETE.json',dict(read_only_audit_complete=True,missing_items_reported=True));notice('MODULE4_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception:dump(RUN/'status/FORENSICS_ERROR.json',dict(error=traceback.format_exc()));raise
