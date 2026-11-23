"""Fail-closed v1.5.2 acceptance checks and immutable bundle construction."""
from micro_common import *
import zipfile,scipy,sklearn

REQUIRED=['gate_v152_check.json','module1_temperature.lock.json','module1_solver_disentanglement_12cond.csv','module1_scope_summary.csv','module1_paired_stats_holm.csv','module2_expanded_dev_grid.csv','module2_dev_plateau_summary.json','module2_roleswap_expanded_check.csv','calib_weighted_unit_test.json','module3_calib_query_cluster_ci_12cond.csv','module3_calib_query_cluster_ci_scope.csv','module3_classwise_calibration_44endpoints.csv','module3_pathology_dca_meta.json','module3_pathology_dca_summary.csv','module3_selected_endpoints_dca.csv','module3_reliability_bins_ecgf_merl.csv','figure5_bspc_v152.pdf','figure5_bspc_v152.png','ecgfounder_interface_forensics.json']

def main():
    pin(10);guard()
    assert read(OUT/'calib_weighted_unit_test.json')['passed'] and read(OUT/'calib_weighted_unit_test.json')['max_abs_diff']<=1e-13
    for n in ['MODULE12_COMPLETE.json','MODULE3_COMPLETE.json','MODULE4_COMPLETE.json']:assert (RUN/n).exists(),n
    assert read(OUT/'figure5_layout_check.json')['passed']
    assert sha(RUN/SPEC)==SPEC_SHA
    for n,v in read(RUN/'FIT_CODE_MANIFEST.json').items():assert sha(SCRIPTS/n)==v,n
    for n,v in read(RUN/'POSTPROCESS_CODE_MANIFEST.json').items():assert sha(SCRIPTS/n)==v,n
    before=read(RUN/'HISTORICAL_TREE_BEFORE.json');after=history_snapshot();changed=[k for k in set(before)|set(after) if before.get(k)!=after.get(k)]
    dump(OUT/'historical_readonly_check.json',dict(passed=not changed,files_checked=len(before),changed_paths=changed));assert not changed
    for n in REQUIRED:assert (OUT/n).exists() and (OUT/n).stat().st_size>0,n
    checks=[]
    for p in OUT.glob('*.csv'):
        f=pd.read_csv(p);num=f.select_dtypes(include=[np.number]);allowed=[]
        if p.name=='module1_solver_disentanglement_12cond.csv':
            assert f.loc[f.cohort!='cpsc2018','cpsc2018_matched8_ap'].isna().all();assert f.loc[f.cohort=='cpsc2018','cpsc2018_matched8_ap'].notna().all();allowed=['cpsc2018_matched8_ap']
        if p.name=='module1_paired_stats_holm.csv':
            assert f.loc[f.p_raw.isna(),'constant_difference'].all();allowed=['p_raw','p_holm']
        if p.name=='module3_reliability_bins_ecgf_merl.csv':
            for col in ['mean_predicted_probability','empirical_positive_rate']:assert f.loc[f[col].isna(),'count'].eq(0).all()
            allowed=['mean_predicted_probability','empirical_positive_rate']
        if 'valid_draws' in f:
            assert (f.valid_draws+f.undefined_replicates==f.total_draws).all()
            for col in ['ci_low','ci_high']:assert f.loc[f[col].isna(),'valid_draws'].eq(0).all()
            allowed+=['ci_low','ci_high']
        assert np.isfinite(num.drop(columns=allowed,errors='ignore').to_numpy()).all(),p.name
        checks.append(dict(file=p.name,rows=len(f),intentional_missing_columns=allowed))
    grid=pd.read_csv(OUT/'module2_expanded_dev_grid.csv');assert len(grid)==192 and grid.groupby(['dev_cohort','backbone']).size().eq(48).all()
    assert grid.episodes.eq(20).all()
    ep=pd.read_csv(OUT/'module1_episode_metrics.csv');assert ep.groupby(['backbone','cohort','shot','method','ste_policy','calibration','evaluation_scope']).seed.nunique().eq(20).all()
    raw=pd.read_csv(OUT/'module2_roleswap_episode_metrics.csv');assert raw.groupby(['backbone','cohort','shot','method','evaluation_scope']).seed.nunique().eq(20).all()
    ci=pd.read_csv(OUT/'module3_calib_query_cluster_ci_12cond.csv');assert ci.groupby(['calibration','metric','comparator']).size().eq(12).all()
    import torch
    cpu=next(v.split(':',1)[1].strip() for v in Path('/proc/cpuinfo').read_text().splitlines() if v.startswith('model name'))
    file_manifest={str(p.relative_to(RUN)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='AUDIT_MANIFEST_v1.5.2.json'}
    audit=dict(all_assertions_passed=True,spec_sha256=SPEC_SHA,fit_code_manifest=read(RUN/'FIT_CODE_MANIFEST.json'),postprocess_code_manifest=read(RUN/'POSTPROCESS_CODE_MANIFEST.json'),environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,torch=torch.__version__,CPU=cpu,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),max_worker_count=4,BLAS_threads_per_worker=1),readonly_check=read(OUT/'historical_readonly_check.json'),CSV_checks=checks,artifacts=file_manifest,
      forensics_missing_items=read(OUT/'ecgfounder_interface_forensics.json')['missing_items'],interpretation_notes=read(OUT/'protocol_interpretation_notes.json'),figure_review=read(OUT/'figure_review_status.json') if (OUT/'figure_review_status.json').exists() else dict(independent_agent_review='not yet completed',primary_render_review='required locally'),completed_at_unix=time.time())
    dump(OUT/'AUDIT_MANIFEST_v1.5.2.json',audit)
    text=['# SASM v1.5.2 micro results','',f'Locked spec SHA-256: {SPEC_SHA}','Stage 0 and weighted calibration gates passed. Modules 1-3 and read-only Module 4 evidence gathering completed.','Module 4 explicitly records missing evidence and distinguishes historical filtered probability caches from current core h caches; unknown clipping provenance is not invented.','Module 1 fallback follows the explicit numeric alpha=0.1; the arm name does not mean the archived LP alpha=100/10.','Expanded-grid/role analyses are development-cohort sensitivity analyses, not new untouched external confirmation.','Calibration CIs condition on the 20 fixed support episodes. Cross-cohort pairing reuses the archived draws with saved cohort permutations.','Classifier predictions/models remain under the server fits/ directory and are not all redistributed in this small bundle. Per-episode metrics, grid files, code, logs and checksums are bundled.','No manuscript, supplementary source, BSPC PDF, GitHub release or tag was changed.']
    (RUN/'README_RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    files=[]
    for base in ['results','scripts','logs','grid','module3_conditions']:
        for p in (RUN/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp') and p.name!='postprocess.log':files.append(p)
    files+=[RUN/SPEC,RUN/'README_RESULTS.md',RUN/'FIT_CODE_MANIFEST.json',RUN/'POSTPROCESS_CODE_MANIFEST.json']
    manifest={str(p.relative_to(RUN)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(files))};dump(RUN/'BUNDLE_FILE_MANIFEST.json',dict(files=manifest));files.append(RUN/'BUNDLE_FILE_MANIFEST.json')
    bundle=ROOT/'SASM_v1.5.2_micro_bundle.zip'
    if bundle.exists():raise RuntimeError('Refuse to overwrite an existing bundle; inspect it first')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):z.write(p,str(p.relative_to(RUN)))
    with zipfile.ZipFile(bundle) as z:assert z.testzip() is None
    dump(RUN/'BUNDLE_COMPLETE.json',dict(all_assertions_passed=True,path=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),time=time.time()));notice('BUNDLE_COMPLETE',bytes=bundle.stat().st_size,sha256=sha(bundle))

if __name__=='__main__':
    try:main()
    except Exception:dump(RUN/'status/PACKAGE_ERROR.json',dict(error=traceback.format_exc()));raise
