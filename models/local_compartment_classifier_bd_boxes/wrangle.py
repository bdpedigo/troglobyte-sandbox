# %%


import pickle
from pathlib import Path

import caveclient as cc
import numpy as np
import pandas as pd

from troglobyte.features import CAVEWrangler

client = cc.CAVEclient("minnie65_phase3_v1")

out_path = Path("./troglobyte-sandbox/models/")

model_name = "local_compartment_classifier_bd_boxes"

data_path = Path("./troglobyte-sandbox/data/bounding_box_labels")

files = list(data_path.glob("*.csv"))

# %%
voxel_resolution = np.array([4, 4, 40])


def simple_labeler(label):
    if "axon" in label:
        return "axon"
    elif "dendrite" in label:
        return "dendrite"
    elif "glia" in label:
        return "glia"
    elif "soma" in label:
        return "soma"
    else:
        return np.nan


def axon_labeler(label):
    if label == "uncertain":
        return np.nan
    elif "axon" in label:
        return True
    else:
        return False


dfs = []
for file in files:
    file_label_df = pd.read_csv(file)

    file_label_df.set_index(["bbox_id", "root_id"], inplace=True, drop=False)
    file_label_df["ctr_pt_4x4x40"] = file_label_df["ctr_pt_4x4x40"].apply(
        lambda x: np.array(eval(x.replace("  ", ",").replace(" ", ",")), dtype=int)
    )

    file_label_df["x_nm"] = (
        file_label_df["ctr_pt_4x4x40"].apply(lambda x: x[0]) * voxel_resolution[0]
    )
    file_label_df["y_nm"] = (
        file_label_df["ctr_pt_4x4x40"].apply(lambda x: x[1]) * voxel_resolution[1]
    )
    file_label_df["z_nm"] = (
        file_label_df["ctr_pt_4x4x40"].apply(lambda x: x[2]) * voxel_resolution[2]
    )

    file_label_df["axon_label"] = file_label_df["label"].apply(axon_labeler)

    file_label_df["simple_label"] = file_label_df["label"].apply(simple_labeler)

    dfs.append(file_label_df)

# %%
label_df = pd.concat(dfs)
label_df.to_csv(out_path / model_name / "labels.csv")

# %%


points = label_df[["x_nm", "y_nm", "z_nm"]].values
neighborhood_hops = 5

# set up object
wrangler = CAVEWrangler(client, verbose=10, n_jobs=-1)

# list the objects we are interested in
wrangler.set_objects(label_df.index.get_level_values("root_id"))

# get a 20um bounding box around the points which were classified
wrangler.set_query_boxes_from_points(points, box_width=20_000)

# query the level2 ids for the objects in those bounding boxes
wrangler.query_level2_ids()

# query the level2 shape features for the objects in those bounding boxes
wrangler.query_level2_shape_features()

# query the level2 synapse features for the objects in those bounding boxes
# this uses the object IDs which were input for the synapse query, which may get out
# of date
wrangler.query_level2_synapse_features(method="existing")

# aggregate these features by k-hop neighborhoods in the level2 graph
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"],
    neighborhood_hops=neighborhood_hops,
    drop_self_in_neighborhood=True,
)

with open(out_path / model_name / "wrangler.pkl", mode="bw") as f:
    pickle.dump(wrangler, file=f)

# %%

X_df = wrangler.features_.copy()
X_df = X_df.drop(columns=[col for col in X_df.columns if "rep_coord" in col])
X_df.to_csv(out_path / model_name / "features.csv")
X_df
