# %%
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from skops.io import load
from troglobyte.features import CAVEWrangler

# %%

client = CAVEclient("minnie65_phase3_v1")
client.materialize.version = 1078

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=3)

# %%
df = pd.read_csv(
    "troglobyte-sandbox/data/blood_vessels/segments_per_branch_2024-08-09.csv"
)
target_df = df.set_index("IDs")
target_df.index.name = "root_id"
# target_df = target_df.sample(2)

# %%
box_params = target_df.groupby(["BranchX", "BranchY", "BranchZ"])[
    [
        "PointA_X",
        "PointA_Y",
        "PointA_Z",
        "PointB_X",
        "PointB_Y",
        "PointB_Z",
        "mip_res_X",
        "mip_res_Y",
        "mip_res_Z",
        "BranchTypeName",
    ]
].first()
box_params["box_id"] = np.arange(len(box_params))

# %%
box_params["x_min"] = box_params["PointA_X"] * box_params["mip_res_X"]
box_params["y_min"] = box_params["PointA_Y"] * box_params["mip_res_Y"]
box_params["z_min"] = box_params["PointA_Z"] * box_params["mip_res_Z"]
box_params["x_max"] = box_params["PointB_X"] * box_params["mip_res_X"]
box_params["y_max"] = box_params["PointB_Y"] * box_params["mip_res_Y"]
box_params["z_max"] = box_params["PointB_Z"] * box_params["mip_res_Z"]

# %%
box_params.set_index(["x_min", "y_min", "z_min", "x_max", "y_max", "z_max"])[
    "BranchTypeName"
]

# %%

model = load(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/local_compartment_classifier_bd_boxes.skops"
)

out_path = Path("troglobyte-sandbox/results/vasculature") / "2024-08-09"

# pad_distance = 20_000
pad_distance = 30_000

for i in range(12, len(box_params)):
    lower = box_params.iloc[i][["x_min", "y_min", "z_min"]].values
    upper = box_params.iloc[i][["x_max", "y_max", "z_max"]].values
    og_box = np.array([lower, upper])

    padded_box = og_box.copy()
    padded_box[0] -= np.array(pad_distance)
    padded_box[1] += np.array(pad_distance)

    box_name = str(og_box.astype(int).ravel()).strip("[]").replace(" ", "_")

    query_root_ids = (
        target_df.reset_index()
        .set_index(["BranchX", "BranchY", "BranchZ"])
        .loc[box_params.index[i]]["root_id"]
    ).values

    currtime = time.time()

    wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=3)
    wrangler.set_objects(query_root_ids)
    wrangler.set_query_box(padded_box)
    wrangler.query_level2_ids()

    wrangler.query_level2_shape_features()
    wrangler.query_level2_synapse_features(method="existing")
    wrangler.query_level2_edges(warn_on_missing=False)
    wrangler.register_model(model, "bd_boxes")
    wrangler.aggregate_features_by_neighborhood(
        aggregations=["mean", "std"], neighborhood_hops=5
    )
    wrangler.stack_model_predict_proba("bd_boxes")

    features = wrangler.features_
    features.to_csv(out_path / f"vasculature_features_box={box_name}.csv")

    with open(out_path / f"wrangler_box={box_name}.pkl", "wb") as f:
        wrangler.client = None
        pickle.dump(wrangler, f)
        wrangler.client = client

print(f"{time.time() - currtime:.3f} seconds elapsed.")
