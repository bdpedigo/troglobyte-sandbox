# %%

import numpy as np
import pandas as pd
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")


# %%
df = pd.read_csv("troglobyte-sandbox/data/blood_vessels/segments_per_branch2.csv")
target_df = df.set_index("IDs")
target_df.index.name = "root_id"

# %%
box_params = target_df.groupby(["X", "Y", "Z"])[
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
    ]
].first()
box_params["box_id"] = np.arange(len(box_params))

# scale boxes by MIP
box_params["x_min"] = box_params["PointA_X"] * box_params["mip_res_X"]
box_params["y_min"] = box_params["PointA_Y"] * box_params["mip_res_Y"]
box_params["z_min"] = box_params["PointA_Z"] * box_params["mip_res_Z"]
box_params["x_max"] = box_params["PointB_X"] * box_params["mip_res_X"]
box_params["y_max"] = box_params["PointB_Y"] * box_params["mip_res_Y"]
box_params["z_max"] = box_params["PointB_Z"] * box_params["mip_res_Z"]

box_params

# %%

# pad_distance = 40_000

i = 0
query_root_ids = (
    target_df.reset_index()
    .set_index(["X", "Y", "Z"])
    .loc[box_params.index[i]]["root_id"]
).values
box_params.iloc[i]

segmentation_resolution = np.asarray(client.chunkedgraph.base_resolution)

box = box_params.iloc[i][["x_min", "y_min", "z_min", "x_max", "y_max", "z_max"]].values
box = box.reshape((2, 3))
box = box / segmentation_resolution
box = box.astype(int)
box = box.T


out = client.chunkedgraph.get_leaves(
    query_root_ids[0],
    stop_layer=2,
    bounds=box,
)

print(len(out))

print(query_root_ids[0])

# %%
