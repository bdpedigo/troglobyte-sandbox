# %%
import time

import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from nglui import statebuilder
from skops.io import load

from troglobyte.features import CAVEWrangler
from troglobyte.features.wrangler import _make_bounding_box_from_point

# %%

client = CAVEclient("minnie65_phase3_v1")
client.materialize.version = 943

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=3)

model = load(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/local_compartment_classifier_bd_boxes.skops"
)

# %%

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

target_df = target_df.sample(10_000)

currtime = time.time()

wrangler.set_objects(target_df["pt_root_id"])
wrangler.set_query_boxes_from_points(points, box_width=80000)
wrangler.query_level2_ids()
# drop some objects that don't have anything in the box
wrangler.object_ids_ = pd.Index(wrangler.manifest_["object_id"].unique())
wrangler.query_level2_shape_features()
wrangler.prune_query_to_box()
wrangler.query_level2_synapse_features(method="existing")
wrangler.query_level2_edges(warn_on_missing=False)
wrangler.register_model(model, "bd_boxes")
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
wrangler.features_.to_csv(
    "troglobyte-sandbox/data/blood_vessels/bd_vasculature_features.csv"
)

# %%
features = wrangler.features_.dropna()

# %%
features.to_csv("troglobyte-sandbox/data/blood_vessels/bd_vasculature_features.csv")

# %%
features[["bd_boxes_axon", "bd_boxes_soma", "bd_boxes_dendrite", "bd_boxes_glia"]]

# %%
(features["bd_boxes_dendrite"] > 0.9).mean()

#%% 

features[['bd_boxes_axon_neighbor_mean', 'bd_boxes_axon']].corr()

# %%
object_posteriors = features.groupby(["object_id"])[
    [
        "bd_boxes_axon_neighbor_mean",
        "bd_boxes_soma_neighbor_mean",
        "bd_boxes_dendrite_neighbor_mean",
        "bd_boxes_glia_neighbor_mean",
    ]
].mean()

threshold = 0.9
object_predictions = (
    (object_posteriors[object_posteriors > threshold])
    .dropna(how="all", axis=0)
    .idxmax(axis=1)
    .str.replace("_neighbor_mean", "")
    .str.replace("bd_boxes_", "")
)

# %%
object_predictions.value_counts()

# %%
query_point = np.array(eval(points.apply(str).unique()[0].replace(" ", ",")))


box = _make_bounding_box_from_point(query_point, 80000)

# %%

sbs = []
dfs = []
img_layer, seg_layer = statebuilder.helpers.from_client(client)

box_mapper = statebuilder.BoundingBoxMapper(
    point_column_a="point_a", point_column_b="point_b"
)
box_layer = statebuilder.AnnotationLayerConfig(name="box", mapping_rules=box_mapper)
box_df = pd.DataFrame(
    {
        "point_a": [box[0] / np.array([8, 8, 40])],
        "point_b": [box[1] / np.array([8, 8, 40])],
    }
)
sbs.append(statebuilder.StateBuilder(layers=[img_layer, seg_layer, box_layer]))
dfs.append(box_df)

sample_object_predictions = object_predictions.sample(20)


colors = sns.color_palette("pastel", n_colors=4).as_hex()
for i, (label, ids) in enumerate(object_predictions.groupby(object_predictions)):
    ids = ids.sample(min(1, len(ids)))
    sub_df = ids.to_frame().reset_index()
    sub_df["color"] = colors[i]
    new_seg_layer = statebuilder.SegmentationLayerConfig(
        source=client.info.segmentation_source(),
        name=label,
        selected_ids_column="object_id",
        color_column="color",
    )
    sbs.append(statebuilder.StateBuilder(layers=[new_seg_layer]))
    dfs.append(sub_df)

sb = statebuilder.ChainedStateBuilder(sbs)

sb.render_state(dfs, return_as="html")

#%%
with open("stash_wrangler_bd_blood_vessels.pkl", "wb") as f:
    import pickle
    wrangler.client = None
    pickle.dump(wrangler, f)
    wrangler.client = client

#%%
