"""Read exported numeric arrays without pickle deserialization."""
from pathlib import Path
import json
import numpy as np


def load_cohort(root,backbone,cohort):
    root=Path(root)
    meta=json.loads((root/"cohorts"/cohort/"metadata.json").read_text(encoding="utf8"))
    cache=json.loads((root/"features"/backbone/cohort/"index.json").read_text(encoding="utf8"))
    chunks=[]
    for name in cache["shards"]:
        with np.load(root/"features"/backbone/cohort/name,allow_pickle=False) as z:
            chunks.append(z["features"])
    features=np.concatenate(chunks,axis=0)
    with np.load(root/"source_heads"/backbone/cohort/"aligned_raw_heads.npz",allow_pickle=False) as z:
        weights=z["raw_weights"];biases=z["raw_biases"]
    with np.load(root/"cohorts"/cohort/"query.npz",allow_pickle=False) as z:
        query=z["indices"];query_y=z["labels"]
    if cohort=="cpsc2018":
        if meta.get("protocol_id")!="cpsc2018-nineclass-20261117-NORM" or features.shape[0]!=6877 or query_y.shape!=(4122,9) or weights.shape[-1]!=9:
            raise ValueError("CPSC2018 cache/protocol mismatch: extract the complete v1.4.0 assets")
        if cache.get("protocol_id")!=meta["protocol_id"]:
            raise ValueError("mixed CPSC2018 feature protocol")
    if features.shape[0]!=meta["released_records"]:
        raise ValueError("feature record count mismatch")
    return features,weights,biases,query,query_y,meta
