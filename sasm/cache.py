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
    assert features.shape[0]==meta["released_records"]
    return features,weights,biases,query,query_y,meta
