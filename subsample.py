import numpy as np
import pandas as pd


select = {
    "SN_Ia": 300,
    "CCSN": 160,
    "SN_Iax": 10,
    # "SLSN-I": 5,
    # "TDE": 5,
    # "ILOT": 5,
    # "KN": 3,
    # "PISN": 3,
    # "AGN": 5,
}

OBJECT_FILE = "data_release/hourglass_objects.parquet"
SPEC_FILE = "data_release/hourglass_spectra.parquet"

objs = pd.read_parquet(OBJECT_FILE)
# spec = pd.read_parquet(SPEC_FILE)

for key in select.keys():
    obj_hold = []
    obj_hold = objs[objs["class"] == key]
    prism_mask = []
    for _, row in obj_hold.iterrows():
        prism_mask.append("PRISM" in row["field"])
    obj_hold = obj_hold[prism_mask]

    if len(obj_hold) < select[key]:
        print(f"Issue with {key}. It only has {len(obj_hold)} objects with prism.")
    else:
        subselection = np.random.choice(obj_hold["cid"], select[key], replace=False)
        print(f"CIDs for {key}")
        print(subselection)
