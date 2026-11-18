from ablation_common import *
import traceback,fcntl

def run(j,F,heads):
 p=output(j);p.parent.mkdir(parents=True,exist_ok=True);ep,si,qi,ys,mu,sd,xs,xq=input_arrays(j,F);ds=j['dataset'];v=np.load(OLD/'vault'/ds/f'{phase(ds)}.npz',allow_pickle=True);assert np.array_equal(v['indices'],qi)
 tab=pd.read_csv(OLD/'data'/ds/'manifest.csv',dtype={'record_id':str,'patient_id':str});assert tab.iloc[si].record_id.tolist()==ep['support_records'];assert tab.iloc[qi].record_id.tolist()==v['records'].astype(str).tolist();assert tab.iloc[qi].patient_id.tolist()==v['patients'].astype(str).tolist()
 cfg=read(ROOT/'protocol.lock.json')['selected'][j['backbone']];meta=read(OLD/'data'/ds/'metadata.json');results={};diags=[]
 if j['variant']=='ensemble':
  src=original_result(j['backbone'],ds,j['shot'],j['seed']);m=read(src);z=np.load(src.with_suffix('.npz'));wp=src.with_suffix('.weights.npz');wz=np.load(wp)
  assert sha(wp)==m['model_sha256'] and sha(src.with_suffix('.npz'))==m['prediction_sha256'];assert m['episode_sha256']==sha(episode(ds,j['shot'],j['seed']));assert np.array_equal(si,z['support_indices']) and np.array_equal(qi,z['query_indices']);assert m['job']['config']==cfg
  w=wz['weight'];b=wz['bias'];assert len(w)==len(b)==10 and np.allclose(mu,wz['scaler_mean'],rtol=0,atol=1e-12) and np.allclose(sd,wz['scaler_sd'],rtol=0,atol=1e-12)
  scores=[];average=np.zeros_like(z['logits'],dtype=float)
  for h in range(10):
   pred=xq@w[h]+b[h];average+=pred/10;sc=metric(v['y'],pred);assert sc['ap'] is not None;results[f'head{h:02d}']=sc;scores.append(sc)
  err=float(abs(average-z['logits']).max());assert err<1e-9,err
  results['SASM-Ens']=metric(v['y'],z['logits']);results['SASM-Single-ep00']=scores[0];results['Mean-single-head']={name:float(np.mean([s[name] for s in scores])) for name in ['ap','auc']}
  for name in ['ap_per_label','auc_per_label']:results['Mean-single-head'][name]=np.mean([s[name] for s in scores],axis=0).tolist()
  checks=dict(ensemble_max_error=err,source_result_sha256=sha(src),source_model_sha256=m['model_sha256'],source_prediction_sha256=m['prediction_sha256'],head_ids=list(range(10)),source_path=str(src))
 else:
  aa=anchors(heads,mu,sd,j['variant'],ds);weights=[];bias=[];pred=np.zeros((len(qi),len(meta['labels'])),float);nonneg=j['variant']=='nonnegative'
  for a,bp in aa:
   w,b,d=solve(xs,ys,cfg['alpha'],cfg['lam'],a,bp,nonneg);g=gradient_check(xs,ys,w,b,a,cfg['alpha'],cfg['lam'],nonneg);d['recomputed_projected_gradient']=g;weights.append(w);bias.append(b);diags.append(d);pred+=(xq@w+b)/10
  wp=p.with_suffix('.weights.npz');np.savez_compressed(wp,weight=np.array(weights),bias=np.array(bias),scaler_mean=mu,scaler_sd=sd);zp=p.with_suffix('.npz');np.savez_compressed(zp,logits=pred,support_indices=si,query_indices=qi)
  results[j['variant']]=metric(v['y'],pred);assert results[j['variant']]['ap'] is not None
  checks=dict(model_sha256=sha(wp),prediction_sha256=sha(zp),min_weight=float(np.min(weights)),full_gradient_verified=True)
 dump(p,dict(job=j,metrics=results,checks=checks,diagnostics=diags,episode_sha256=sha(episode(ds,j['shot'],j['seed'])),protocol_sha256=sha(ROOT/'protocol.lock.json'),labels=meta['labels'],query_identity_verified=True,time=time.time(),pid=os.getpid()))

def main(name):
 lock=(ROOT/'workers'/f'{name}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ctx={};srcs={}
 while not (ROOT/'ANALYSIS_COMPLETE.json').exists():
  did=False
  for j in jobs():
   if output(j).exists():continue
   claim=ROOT/'claims'/taskkey(j)
   try:claim.mkdir()
   except FileExistsError:continue
   dump(claim/'owner.json',dict(pid=os.getpid(),worker=name,job=j,time=time.time()));dump(ROOT/'workers'/f'{name}.json',dict(pid=os.getpid(),time=time.time(),stage='running',job=j))
   ds=j['dataset'];bb=j['backbone'];idx=(bb,ds)
   if idx not in ctx:ctx[idx]=context(ds,bb)
   if ds not in srcs and j['variant']!='ensemble':srcs[ds]=source_context(ds)
   run(j,ctx[idx],srcs.get(ds));did=True;break
  if not did:time.sleep(10)
 dump(ROOT/'workers'/f'{name}.json',dict(pid=os.getpid(),time=time.time(),stage='complete'))
if __name__=='__main__':
 try:main(sys.argv[1])
 except Exception:dump(ROOT/'errors'/f'{sys.argv[1]}_{os.getpid()}.json',dict(time=time.time(),error=traceback.format_exc()));raise
