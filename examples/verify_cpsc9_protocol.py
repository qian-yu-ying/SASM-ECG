"""Fail closed on mixed CPSC2018 mappings, episodes, groups or caches."""
from pathlib import Path
import sys,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sasm.cache import load_cohort

def main():
    root=ROOT/'data';co=root/'cohorts/cpsc2018'
    z=np.load(co/'labels_and_roles.npz',allow_pickle=False);y=z['labels'];qi=z['query_indices'];pool=z['support_pool']
    if y.shape!=(6877,9) or len(qi)!=4122 or len(pool)!=2755 or set(qi)&set(pool):raise ValueError('nine-class roles')
    with (co/'manifest.csv').open(encoding='utf-8',newline='') as f:manifest=list(csv.DictReader(f))
    groups=np.array([r['split_group'] for r in manifest])
    if set(groups[pool])&set(groups[qi]):raise ValueError('waveform-group overlap')
    if len(set(groups[qi]))!=3966:raise ValueError('query group count')
    m=np.load(root/'mappings/cpsc2018.npz',allow_pickle=False);q=m['concept_map']
    if q.shape!=(71,9) or not np.array_equal(q.sum(0),[1]*8+[0]):raise ValueError('concept coverage')
    for bb in ['ecgfounder','merl']:
        F,rw,rb,query,qy,meta=load_cohort(root,bb,'cpsc2018')
        if not np.array_equal(query,qi) or not np.array_equal(qy,y[qi]):raise ValueError('query labels/order')
        if np.any(rw[:,:,-1]) or np.any(rb[:,-1]):raise ValueError('nonzero STE source prior')
        if meta['anchor_lambda_multiplier']!=[1]*8+[0]:raise ValueError('STE penalty')
        if not np.isfinite(F).all():raise ValueError('nonfinite features')
    for shot in [5,10]:
        for seed in range(20):
            ep=json.loads((co/f'episodes/shot{shot}_seed{seed:02d}.json').read_text())
            si=np.asarray(ep['support'],int)
            if not set(si)<=set(pool) or set(groups[si])&set(groups[qi]):raise ValueError('episode contamination')
            if not np.array_equal(ep['query'],qi) or not np.array_equal(ep['support_y'],y[si]):raise ValueError('episode labels')
    print(json.dumps(dict(passed=True,records=6877,classes=9,support_pool=2755,query_records=4122,query_waveform_groups=3966,episodes=40,source_matched=8,native_shared=7,patient_independence_claimed=False)))
if __name__=='__main__':main()
