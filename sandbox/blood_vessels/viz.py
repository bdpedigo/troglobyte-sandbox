# %%
import pickle

from caveclient import CAVEclient

from troglobyte.features import CAVEWrangler

client = CAVEclient("minnie65_phase3_v1")

wrangler_path = (
    "troglobyte-sandbox/sandbox/blood_vessels/stash_wrangler_bd_blood_vessels.pkl"
)

wrangler: CAVEWrangler = pickle.load(open(wrangler_path, "rb"))
wrangler.client = client
# %%
wrangler

# %%
wrangler.sample_objects(n=100)

# %%
wrangler.query_level2_networks(validate=False)

#%%

import pandas as pd 
import time 
import numpy as np 
from skops.io import load

model = load(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/local_compartment_classifier_bd_boxes.skops"
)

target_df = pd.read_csv("troglobyte-sandbox/data/blood_vessels/segments_per_branch.csv")
target_df.rename(columns={"IDs": "pt_root_id"}, inplace=True)
target_df["bv_position"] = target_df[["X", "Y", "Z"]].apply(np.array, axis=1)
points = target_df.set_index("pt_root_id")["bv_position"]
points = points.apply(lambda x: x * np.array([4, 4, 40]))

target_df = target_df.sample(100)

currtime = time.time()

wrangler.set_objects(target_df["pt_root_id"])
wrangler.set_query_boxes_from_points(points, box_width=80000)
wrangler.query_level2_ids()
# drop some objects that don't have anything in the box
wrangler.object_ids_ = pd.Index(wrangler.manifest_["object_id"].unique())
wrangler.query_level2_shape_features()
wrangler.prune_query_to_boxes()
wrangler.query_level2_synapse_features(method="existing")
wrangler.query_level2_edges(warn_on_missing=False)
wrangler.register_model(model, "bd_boxes")
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5, aggregate_predictions=False
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

#%%
