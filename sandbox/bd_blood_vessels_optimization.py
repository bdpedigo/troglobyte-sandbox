# %%
import time

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from skops.io import load

from troglobyte.features import CAVEWrangler

# %%

client = CAVEclient("minnie65_phase3_v1")
client.materialize.version = 943

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=3)

model = load(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/local_compartment_classifier_bd_boxes.skops"
)

# target_df = pd.read_csv(
#     "troglobyte-sandbox/data/blood_vessels/bd_vasculature_segments_to_classify.csv"
# )
# points = (
#     target_df.set_index("pt_root_id")["bv_position"]
#     .apply(lambda x: x[1:-1].split(","))
#     .apply(lambda x: np.array([float(i) for i in x]) * np.array([4, 4, 40]))
# )


target_df = pd.read_csv("troglobyte-sandbox/data/blood_vessels/segments_per_branch.csv")
target_df.rename(columns={"IDs": "pt_root_id"}, inplace=True)
target_df["bv_position"] = target_df[["X", "Y", "Z"]].apply(np.array, axis=1)
points = target_df.set_index("pt_root_id")["bv_position"]
points = points.apply(lambda x: x * np.array([4, 4, 40]))


currtime = time.time()

wrangler.set_objects(target_df["pt_root_id"])
wrangler.set_query_boxes_from_points(points, box_width=80000)
wrangler.query_level2_ids()

# %%
target_df = target_df.sample(500)

currtime = time.time()
wrangler2 = CAVEWrangler(client=client, n_jobs=-1, verbose=3)
wrangler2.set_objects(target_df["pt_root_id"])
wrangler2.set_query_boxes_from_points(points, box_width=80000)
wrangler2.query_level2_edges(warn_on_missing=False)
wrangler2.query_level2_ids_from_edges()
wrangler2.query_level2_shape_features()
wrangler2.query_level2_synapse_features(method="existing")
wrangler2.object_ids_ = pd.Index(wrangler2.manifest_["object_id"].unique())
wrangler2.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5
)
wrangler2.register_model(model, "bd_boxes")
wrangler2.stack_model_predict_proba("bd_boxes")
wrangler2.stack_model_predict("bd_boxes")
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
# drop some objects that don't have anything in the box
wrangler.object_ids_ = pd.Index(wrangler.manifest_["object_id"].unique())
wrangler.query_level2_shape_features()
# wrangler.prune_query_to_box()
wrangler.query_level2_synapse_features(method="existing")
wrangler.query_level2_edges(warn_on_missing=False)
wrangler.register_model(model, "bd_boxes")

# %%

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(6, 5))
sns.histplot(
    wrangler2.features_.dropna().groupby("object_id")["size_nm3"].sum(),
    log_scale=(True, False),
    ax=ax,
)

sizes = wrangler2.features_.dropna().groupby("object_id")["size_nm3"].sum()

#%%
sizes.quantile(0.1)


# %%
import yappi

wrangler.n_jobs = 1
yappi.clear_stats()
yappi.set_clock_type("cpu")
yappi.start()
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5
)
yappi.get_func_stats().print_all()
yappi.stop()

print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
