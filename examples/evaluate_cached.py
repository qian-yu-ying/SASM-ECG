"""Evaluate existing cached episodes; no feature extraction or source training."""
import os
for key in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key,"1")
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from threadpoolctl import threadpool_limits
from sasm.cache import load_cohort
from sasm.model import fit_cached_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",default=str(Path(__file__).resolve().parents[1]/"data"))
    p.add_argument("--backbone",choices=["ecgfounder","merl"],default="ecgfounder")
    p.add_argument("--cohort",choices=["ningbo","cpsc2018","sph"],default="cpsc2018")
    p.add_argument("--shot",type=int,choices=[5,10],default=5)
    p.add_argument("--seeds",default="0",help="comma-separated seed IDs, or all for the 20-seed mean")
    p.add_argument("--methods",default="sasm-ens,meananchor,lp")
    p.add_argument("--output",type=Path)
    args=p.parse_args();root=Path(args.data)
    seeds=list(range(20)) if args.seeds=="all" else [int(x) for x in args.seeds.split(",")]
    if not seeds or any(s not in range(20) for s in seeds) or len(set(seeds))!=len(seeds):p.error("seed IDs must be distinct integers from 0 to 19")
    methods=args.methods.split(",")
    if any(m not in {"sasm-ens","meananchor","lp"} for m in methods):p.error("invalid method")
    features,rw,rb,query,qy,meta=load_cohort(root,args.backbone,args.cohort)
    cfg=json.loads((root/"configurations.json").read_text(encoding="utf8"))[args.backbone]
    rows=[]
    with threadpool_limits(limits=1):
        for seed in seeds:
            episode=json.loads((root/"cohorts"/args.cohort/"episodes"/f"shot{args.shot}_seed{seed:02d}.json").read_text(encoding="utf8"))
            support=np.asarray(episode["support"],dtype=int);sy=np.asarray(episode["support_y"],dtype=float)
            assert not np.intersect1d(support,query).size
            for method in methods:
                params=cfg["lp"] if method=="lp" else cfg["sasm"]
                start=time.perf_counter()
                w,b,diags=fit_cached_features(features[support],sy,rw,rb,method=method,**params)
                seconds=time.perf_counter()-start
                scores=features[query].astype(float)@w+b
                row=dict(seed=seed,method=method,macro_ap=100*float(average_precision_score(qy,scores,average="macro")),macro_auc=100*float(roc_auc_score(qy,scores,average="macro")),fit_seconds=seconds,max_gradient=max(d["max_gradient"] for d in diags))
                rows.append(row);print(json.dumps(row),flush=True)
    summaries=[]
    refs=json.loads((root/"paper_reference.json").read_text(encoding="utf8"))
    for method in methods:
        group=[r for r in rows if r["method"]==method]
        s=dict(method=method,n_seeds=len(group),macro_ap_mean=float(np.mean([x["macro_ap"] for x in group])),macro_auc_mean=float(np.mean([x["macro_auc"] for x in group])))
        if sorted(seeds)==list(range(20)):
            s["paper_rounded_reference"]=refs[args.backbone][args.cohort][str(args.shot)][method]
        summaries.append(s)
    result=dict(backbone=args.backbone,cohort=args.cohort,shot=args.shot,rows=rows,summary=summaries)
    print(json.dumps({"summary":summaries},indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding="utf8")


if __name__=="__main__":main()
