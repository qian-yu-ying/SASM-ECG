from ablation_common import *
import subprocess,shutil,platform

def main():
 assert not (ROOT/'launch.json').exists(),'Already launched; inspect actual PIDs before any recovery'
 assert (SUP/'MAIN_COMPLETE.json').exists();gate=read(ROOT/'prerequisite_gate.json');assert sha(SUP/'MAIN_COMPLETE.json')==gate['predecessor_complete_sha256'];assert (SUP/'LOCAL_MAIN_REPORTS_VERIFIED.json').exists()
 for name in ['workers','claims','errors','logs','results','tmp','cache','timing_supports']:(ROOT/name).mkdir(exist_ok=True)
 original_protocol=read(OLD/'protocol.lock.json')
 for name,h in original_protocol['source_hashes'].items():assert sha(OLD/'source/source_heads'/name)==h
 for name,h in original_protocol['asset_hashes'].items():assert sha(OLD/'source'/name)==h
 for cfg in [read(SUP/'merl/protocol.lock.json')]:
  for asset in [cfg['checkpoint'],cfg['architecture'],cfg['source_protocol'],*cfg['source_heads']]:assert sha(asset['path'])==asset['sha256']
 selected=dict(ECGFounder=read(OLD/'release_01.json')['selected']['SASM-Ens']['config'],MERL=read(SUP/'merl/release.json')['selected']['SASM-Ens']['config']);perms={}
 for dsindex,ds in enumerate(DS):
  heads=source_context(ds);n=heads[0]['raw'].shape[1];valid=np.flatnonzero(np.any(np.stack([h['raw'] for h in heads])!=0,axis=(0,1)));perms[ds]={}
  for variant,fraction in [('perm25',.25),('perm50',.5)]:
   rng=np.random.default_rng(20261103+['ningbo','chapman','cpsc2018','sph'].index(ds)*100+(25 if variant=='perm25' else 50));m=len(valid);desired=int(np.floor(m*fraction+.5));count=0 if m<2 else min(m,max(2,desired));order=rng.permutation(valid);chosen=order[:count];mapping=np.arange(n)
   if count:mapping[chosen]=np.roll(chosen,1)
   assert np.sum(mapping!=np.arange(n))==count and sorted(mapping.tolist())==list(range(n));assert not set(np.flatnonzero(mapping!=np.arange(n)))-set(valid)
   perms[ds][variant]=dict(nominal_fraction=fraction,nonzero_columns=valid.tolist(),mismatched_columns=chosen.tolist(),mismatched_count=count,achieved_fraction=count/m if m else None,mapping=mapping.tolist(),rng_seed=20261103+['ningbo','chapman','cpsc2018','sph'].index(ds)*100+(25 if variant=='perm25' else 50),rule='round nearest nominal*m; if one, choose two because one-column permutation impossible; rotate seeded subset; fixed per dataset')
 assert perms=={ds:read(BASE.parent/'ensemble_ablation_20261103/permutations.json')[ds] for ds in DS},'Historical fixed mappings must be unchanged'
 dump(ROOT/'permutations.json',perms)
 scripts=['ablation_common.py','run_ablation.py','supervisor.py','timing.py','bootstrap.py']
 protocol=dict(created=time.time(),prerequisite_gate=gate,selected=selected,datasets=DS,shots=[5,10],seeds=list(range(20)),jobs=960,code_hashes={p:sha(ROOT/p) for p in scripts},original_release_sha256=sha(OLD/'release_01.json'),merl_release_sha256=sha(SUP/'merl/release.json'),original_protocol_sha256=sha(OLD/'protocol.lock.json'),source_assets={str(p):sha(p) for p in [OLD/'cpu_worker.py',OLD/'core.py',OLD/'source/target_contract.py',OLD/'assets/dense.npz',*[OLD/'source/source_heads'/f'ep{e:02d}_bce_signed_model.npz' for e in range(10)]]},original_feature_seals={ds:read(OLD/'data'/ds/'FEATURES_READY.json') for ds in DS},merl_feature_seals={ds:read(SUP/'merl/data'/ds/'FEATURES_READY.json') for ds in DS},manifests={ds:sha(OLD/'data'/ds/'manifest.csv') for ds in DS},permutations_sha256=sha(ROOT/'permutations.json'),source_head_ids=list(range(10)),ensemble='mean logits; verify reproduced original<1e-9',single_reference='head00 prespecified; primary difference versus mean of10 individual-head metrics',variants=VARIANTS,scale_formulas=dict(full='diag(target_sd) Dw.T diag(1/source_sd) Ws',source_only='Dw.T diag(1/source_sd) Ws',target_only='diag(target_sd) Dw.T Ws',none='Dw.T Ws'),bias_prior_rule='fixed full-conversion source affine bias except permutation moves matched offsets; prior affects constant labels only; active intercept free',constant_rule='same as current main fallback, with nonnegative constant weights nearest feasible positive-part anchor; never clip optimized active-label weights',nonnegative='all active target weight entries>=0, free intercept; genuine L-BFGS-B constraints and projected-gradientKKT<=5e-5',statistics=dict(unit='20 paired target support seeds, conditional fixed query and ten-source-head library',bootstrap=10000,seed=20261103,wilcoxon='two-sided zero_method wilcox method auto correction false; allzero p1',holm='per backbone/metric/family; ensemble12,scale18,constraint6,mismatch12 external comparisons; all three external cohorts including Ningbo',patient_population_claims=False),timing=dict(shots=[5,10,15],seeds=list(range(5)),repetitions=3,warmups=1,threads=1,affinity='one allowed CPU',same_cpu_all_methods=True),retrospective=True,source_reference_sha256=sha(ROOT/'historical_scale_reference.py'))
 dump(ROOT/'protocol.lock.json',protocol)
 # Smoke checks without consuming an evaluation episode: actual constrained solver, KKT, shape/order and aggregate identity.
 rng=np.random.default_rng(1703);x=rng.normal(size=(16,5));y=(rng.random((16,3))>.5).astype(float);a=rng.normal(size=(5,3));w,b,d=solve(x,y,1.,10.,a,np.zeros(3),True);g=gradient_check(x,y,w,b,a,1.,10.,True);assert w.min()>=0
 for ds in DS:
  j=dict(backbone='ECGFounder',variant='ensemble',dataset=ds,shot=5,seed=0);p=original_result('ECGFounder',ds,5,0);m=read(p);wp=p.with_suffix('.weights.npz');assert sha(wp)==m['model_sha256'];z=np.load(wp);assert z['weight'].shape[0]==10
 dump(ROOT/'SMOKE_PASS.json',dict(time=time.time(),toy_nonnegative_kkt=g,toy_min_weight=float(w.min()),threads=1,protocol_sha256=sha(ROOT/'protocol.lock.json')))
 env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'tmp'),TEMP=str(ROOT/'tmp'),TMP=str(ROOT/'tmp'),CUDA_CACHE_PATH=str(ROOT/'cache'),WANDB_MODE='disabled');launched=[]
 for n in range(40):
  name=f'worker{n:02d}'
  with (ROOT/'logs'/f'{name}.log').open('ab') as f:p=subprocess.Popen([sys.executable,str(ROOT/'run_ablation.py'),name],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  launched.append(dict(name=name,pid=p.pid,args=['run_ablation.py',name]))
 with (ROOT/'logs/supervisor.log').open('ab') as f:p=subprocess.Popen([sys.executable,str(ROOT/'supervisor.py')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 launched.append(dict(name='supervisor',pid=p.pid,args=['supervisor.py']));dump(ROOT/'launch.json',dict(time=time.time(),processes=launched,protocol_sha256=sha(ROOT/'protocol.lock.json')));print(json.dumps(launched))
if __name__=='__main__':main()
