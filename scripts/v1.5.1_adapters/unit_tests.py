from adapters import *

def main():
    pin(12);assert read(OUT/'sanity_gate_v1.5.1_check.json')['all_passed'];lp_unit_test()
    rng=np.random.default_rng(151);n,d,c=9,7,3
    x=rng.normal(size=(n,d));a=rng.normal(size=(d,c));pb=rng.normal(size=c);y=rng.integers(0,2,(n,c)).astype(float)
    ep=dict(seed=0,x=x,ss=x@a+pb,a=a,pb=pb,y=y,matched=np.ones(c,bool))
    cfgs=[dict(gamma=.2,lambda_res=.1,lr=.001,epochs=20),dict(gamma=1.,lambda_res=1.,lr=.005,epochs=20)]
    models=fit_task_batch(ep,cfgs);worst=0.
    for cfg,model in zip(cfgs,models):
        w=torch.nn.Parameter(torch.zeros((d,c),dtype=torch.float64));b=torch.nn.Parameter(torch.zeros(c,dtype=torch.float64))
        opt=torch.optim.Adam([w,b],lr=cfg['lr'],weight_decay=0.);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['epochs'])
        tx=torch.tensor(x);ty=torch.tensor(y);ta=torch.tensor(a);tb=torch.tensor(pb)
        for _ in range(cfg['epochs']):
            opt.zero_grad(set_to_none=True);z=tx@(ta+cfg['gamma']*w)+tb+b
            loss=TF.binary_cross_entropy_with_logits(z,ty,reduction='sum')/n+cfg['lambda_res']*w.square().sum()/n
            loss.backward();opt.step();sched.step()
        expected=x@(a+cfg['gamma']*w.detach().numpy())+pb+b.detach().numpy()
        actual=x@model['weight']+model['bias'];worst=max(worst,float(abs(actual-expected).max()))
    assert worst<1e-11
    delta=dict(methods={m:dict(multilabel_BCE=True,frozen_waveform_features=True,text_encoder_used=False,
                 prior='PTBXL ten-head mean coordinate-converted source prior',changes_from_original=['CLIP image/text interface replaced with frozen ECG and supervised source-head interface','multiclass objective/readout replaced by independently specified multilabel ECG/BCE extension']) for m in METHODS},
        individual={METHODS[0]:dict(normalized_cache=True,encodings=['signed','positive'],STE='LP plus cache',configs=144),
                    METHODS[1]:dict(trainable_cache_keys=True,trainable_bias=True,bias_weight_decay=0.,configs=24,STE='LP plus trained cache'),
                    METHODS[2]:dict(optimizer='torch.optim.Adam',weight_decay=0.,explicit_residual_L2=True,configs=135,STE='zero slope plus Jeffreys bias and learned residual'),
                    METHODS[3]:dict(joint_augmented_Hessian=True,unconstrained_mixing=True,configs=160,STE='alpha fixed zero; no source column')},
        spec_sha256=SPEC_SHA,original_method_fidelity_claim=False,hyperparameter_tie_rule='exact mean development AP descending, canonical config JSON ascending',prior_external_exposure=True)
    dump(OUT/'adapter_protocol_deltas.json',delta)
    dump(OUT/'implementation_unit_tests.json',dict(all_passed=True,taskres_batched_vs_independent_Adam_max_logit_error=worst,
       grid_counts={m:len(grid(m)) for m in METHODS},torch=torch.__version__,dtype='float64',GPU_used=False))
    notice('IMPLEMENTATION_UNIT_PASSED',taskres_error=worst)

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;dump(OUT/'UNIT_ERROR.json',dict(traceback=traceback.format_exc()));raise
