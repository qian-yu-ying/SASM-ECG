"""Fail-closed verification, transparent replay boundary and result bundle."""
from core import *
import zipfile,gzip,subprocess

REQUIRED=['sanity_gate_v1.5.1_check.json','lpplus_monotone_unit_test.json','adapter_protocol_deltas.json','dev_chapman_grid_search.json','dev_sph_roleswap_grid_search.json','main_12cond_summary.csv','main_paired_stats_holm.csv','main_query_cluster_ci.csv','classwise_5shot_ap_44endpoints.csv','translational_calibration_summary.csv','translational_dca_summary.csv','translational_locked_rules_summary.csv','subgroup_48slices_summary.csv','roleswap_sph_12cond_summary.csv','cpu_singlecore_latency.csv','per_seed_raw_records.json.gz']

def main():
    pin(10)
    for n in ['MAIN_FITS_COMPLETE.json','SWAP_FITS_COMPLETE.json','MAIN_SCORING_COMPLETE.json','SWAP_SCORING_COMPLETE.json','TIMING_COMPLETE.json','BOOTSTRAPS_COMPLETE.json']:assert (RUN/n).exists(),n
    assert read(OUT/'sanity_gate_v1.5.1_check.json')['all_passed']
    assert read(OUT/'lpplus_monotone_unit_test.json')['all_monotone']
    assert read(OUT/'implementation_unit_tests.json')['all_passed'] and read(OUT/'postprocess_unit_tests.json')['all_passed']
    assert sha(RUN/'SASM_v1.5.1_BSPC_Adapter_Server_Spec_LOCKED.md')==SPEC_SHA
    for n,v in read(RUN/'FIT_CODE_MANIFEST.json').items():assert sha(SCRIPTS/n)==v,n
    for n,v in read(RUN/'POSTPROCESS_CODE_MANIFEST.json').items():assert sha(SCRIPTS/n)==v,n
    for n in REQUIRED:assert (OUT/n).exists(),n
    checks=[]
    for n in ['main_12cond_summary.csv','translational_calibration_summary.csv','translational_dca_summary.csv','translational_locked_rules_summary.csv','roleswap_sph_12cond_summary.csv','cpu_singlecore_latency.csv','classwise_5shot_ap_44endpoints.csv']:
        f=pd.read_csv(OUT/n);num=f.select_dtypes(include=[np.number]);assert np.isfinite(num.to_numpy()).all(),n
        checks.append(dict(file=n,rows=len(f),all_numeric_finite=True))
    f=pd.read_csv(OUT/'metric_episodes.csv');assert len(f)==5760
    g=f.groupby(['backbone','cohort','shot','method','calibration']);assert g.seed.nunique().eq(20).all() and g.size().eq(20).all()
    roles=pd.read_csv(OUT/'roleswap_episodes.csv');assert len(roles)==1920 and roles.groupby(['backbone','cohort','shot','method']).seed.nunique().eq(20).all()
    for mode in ['main','swap']:
        lock=read(OUT/('dev_chapman_grid_search.json' if mode=='main' else 'dev_sph_roleswap_grid_search.json'));assert lock['all_four_methods_locked']
        assert all(v['episodes']==20 for bb in lock['selected'].values() for v in bb.values())
        assert all(v['episodes']==20 for bb in lock['grid'].values() for method in bb.values() for v in method)
    for n in ['main_query_cluster_ci.csv','subgroup_48slices_summary.csv']:
        cf=pd.read_csv(OUT/n);assert (cf.valid_draws+cf.undefined_replicates==cf.total_draws).all()
        assert np.isfinite(cf.loc[cf.valid_draws>0,['ci_low','ci_high']].to_numpy()).all()
        assert (cf.loc[cf.valid_draws==0,'undefined_reason'].notna()).all()
    assert len(pd.read_csv(OUT/'main_query_cluster_ci.csv'))==288
    assert len(pd.read_csv(OUT/'subgroup_48slices_summary.csv'))==384
    assert len(pd.read_csv(OUT/'cpu_singlecore_latency.csv'))==96
    for fn in [OUT/'main_paired_stats_holm.csv',OUT/'roleswap_paired_stats_holm.csv',OUT/'dca_paired_stats_holm.csv',OUT/'locked_rule_paired_stats_holm.csv']:
        s=pd.read_csv(fn);assert np.isfinite(s[['difference','ci_low','ci_high']].to_numpy()).all()
        assert s.loc[s.p_raw.isna(),'constant_difference'].all()
        assert s.loc[s.p_holm.notna(),'p_holm'].between(0,1).all()
    fitfiles=sorted((RUN/'fits').glob('*/*/*/shot*.npz'));assert len(fitfiles)==560
    assets=[]
    for p in fitfiles:
        meta=read(p.with_suffix('.json'));assert sha(p)==meta['sha256'];assets.append(dict(path=str(p.relative_to(RUN)),bytes=p.stat().st_size,sha256=meta['sha256']))
    dump(OUT/'server_prediction_assets_manifest.json',dict(server_root=str(RUN),assets=assets,in_bundle=False,
      reason='Fitted model/query arrays remain on node4; result bundle includes per-seed metrics, code, selection grids, exact bootstrap draws and checksums. Raw cached ECG features/checkpoints are not redistributed.'))
    with gzip.open(OUT/'per_seed_raw_records.json.gz','rt',encoding='utf-8') as stream:raw=json.load(stream)
    assert len(raw['metric_episodes'])==5760
    import scipy,sklearn,torch
    cpu=next(v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name'))
    audit=dict(all_passed=True,spec_sha256=SPEC_SHA,baseline_release='v1.4.1',baseline_commit='ef4502c75aad736ff2ff7c226067723a75590f5e',
      fit_code_manifest=read(RUN/'FIT_CODE_MANIFEST.json'),postprocess_code_manifest=read(RUN/'POSTPROCESS_CODE_MANIFEST.json'),
      environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,torch=torch.__version__,cpu=cpu,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),max_worker_count=4,BLAS_threads_per_worker=1),
      verification=checks,fit_assets=len(fitfiles),external_episodes_per_role=240,development_episodes_per_role=40,
      independent_agent_validation='not completed; prior external service returned 403 insufficient balance; all recorded tests were executed by primary agent',
      bootstrap=read(OUT/'bootstrap_draws_manifest.json'),metric_units='fractions; multiply by 100 for percentage points; timing seconds',
      replay_boundary='Metrics and resampling auditable from bundled records/code/draws. Full fit replay needs licensed public raw waveforms/checkpoints or server-resident feature caches and historical dependencies. No end-to-end encoder cost claim.',
      completed_at_unix=time.time())
    dump(OUT/'AUDIT_MANIFEST_v1.5.1.json',audit)
    summary=pd.read_csv(OUT/'scope_summary.csv');rank=summary[(summary.scope=='Overall')&(summary.calibration=='Raw')].sort_values('ap',ascending=False)
    txt=['# SASM v1.5.1 execution results','', 'All locked stages completed. Numbers below are equal-weight means across 12 conditions.','',rank[['method','ap','auc','ece','f1']].to_string(index=False),'',
      'These four adapters are explicitly ECG/BCE extensions, not unchanged reproductions of the CLIP methods.','No manuscript or GitHub release was changed. External-query labels were used only for scoring and the explicitly retrospective Oracle Adj/ROC/DCA analyses.',
      'Single-core timings exclude frozen encoder extraction/search/data loading; Tip methods on CPSC reuse the archived LP fallback. See the per-row timing scope.',
      'Query CIs are conditional on the 20 fixed support episodes. SPH uses patient clusters; Ningbo uses released record units, not verified patients; CPSC uses full-waveform groups.',
      'Independent fresh-agent validation did not complete (external 403). Primary-agent numerical witnesses and integrity assertions passed.',
      'Fitted predictions/models stay on node4 and are indexed by server_prediction_assets_manifest.json; they are not contained in this ZIP.']
    (RUN/'RESULTS_README.md').write_text('\n'.join(txt)+'\n',encoding='utf-8')
    include=[]
    for base in ['results','scripts','logs','draws','grid','query_ci','subgroup_ci']:
        for p in (RUN/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp'):include.append(p)
    include += [p for p in RUN.glob('*.json') if p.is_file()]+[RUN/'RESULTS_README.md',RUN/'SASM_v1.5.1_BSPC_Adapter_Server_Spec_LOCKED.md']
    inventory={str(p.relative_to(RUN)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(include)) if p.name!='postprocess.log'}
    dump(RUN/'BUNDLE_FILE_MANIFEST.json',dict(files=inventory,mutable_running_log_excluded=True));include.append(RUN/'BUNDLE_FILE_MANIFEST.json')
    bundle=RUN/'SASM_v1.5.1_adapters_bundle.zip'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(include)):
            if p.name!='postprocess.log':z.write(p,str(p.relative_to(RUN)))
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    dump(RUN/'BUNDLE_COMPLETE.json',dict(all_passed=True,path=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),time=time.time()))
    notice('BUNDLE_COMPLETE',bytes=bundle.stat().st_size,sha256=sha(bundle))

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;dump(RUN/'status'/'PACKAGE_ERROR.json',dict(traceback=traceback.format_exc()));raise
