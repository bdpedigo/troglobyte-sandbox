# %%
from pathlib import Path

import caveclient as cc
import numpy as np
from skops.io import load

from troglobyte.features import CAVEWrangler

client = cc.CAVEclient("minnie65_phase3_v1")

out_path = Path("./troglobyte-sandbox/models/")

model_name = "local_compartment_classifier_bd_boxes"


# %%
# model = load(out_path / model_name / f"{model_name}_no_syn.skops")
model = load(out_path / model_name / f"{model_name}.skops")

# %%
wrangler = CAVEWrangler(client=client, verbose=10, n_jobs=-1)

# %%

# w-q box in white matter
start = np.array([263697, 279652, 21026])
end = np.array([266513, 282468, 21314])

# w-q box new box in white matter
start = np.array([329598, 252614, 20496])
end = np.array([304910, 281450, 23096])
# start = np.array([140252, 231547, 21012])
# end = np.array([143068, 234363, 21300])

import pandas as pd

object_ids = pd.read_csv("wq-ids.csv").columns.str.strip(' ').astype(int)

diff = end - start
box = np.array([start, end])

# %%
wrangler.query_objects_from_box(
    box, source_resolution=np.array([4, 4, 40]), size_threshold=200
)

# define a larger bbox for looking at features
new_start = start - diff
new_end = end + diff
new_box = np.array([new_start, new_end])
wrangler.set_query_box(new_box, source_resolution=np.array([4, 4, 40]))

wrangler.query_level2_edges()

wrangler.query_level2_shape_features()

wrangler.query_level2_synapse_features()

wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5
)

# %%
X_df = wrangler.features_
X_df = X_df.dropna()
X_df = X_df[model.feature_names_in_]

# %%

y_pred = model.predict(X_df)


import pandas as pd

y_pred = pd.Series(y_pred, index=X_df.index, name="compartment_label").to_frame()

# %%
import seaborn as sns

class_to_int_map = {"soma": 1, "axon": 2, "dendrite": 3, "glia": 4, "unknown": 0}

colors = sns.color_palette("tab10", n_colors=5).as_hex()
color_map = {k: colors[i] for i, k in enumerate(class_to_int_map.keys())}

# %%

component_pred_counts = (
    y_pred.groupby("object_id")["compartment_label"].value_counts().unstack().fillna(0)
)
component_pred_probs = component_pred_counts.div(
    component_pred_counts.sum(axis=1), axis=0
)

threshold = 0.8

certain_objects = component_pred_probs[component_pred_probs > threshold].any(axis=1)
certain_objects = certain_objects[certain_objects].index
certain_y_pred = y_pred.loc[certain_objects]

# %%
object_pred = (
    y_pred.groupby("object_id")["compartment_label"]
    .apply(lambda x: x.value_counts().idxmax())
    .to_frame()
)
object_pred["certainty"] = component_pred_probs.max(axis=1)
object_pred["color"] = object_pred["compartment_label"].map(color_map)
object_pred["l2_count"] = y_pred.groupby("object_id").size()

# %%
certain_y_pred = certain_y_pred.reset_index()
certain_y_pred["color"] = certain_y_pred["compartment_label"].map(color_map)

# %%
import json

from nglui import statebuilder

bbox_df = pd.DataFrame(
    {"point_a": [start], "point_b": [end]},
)


sbs = []
dfs = []
layers = []
viewer_resolution = client.info.viewer_resolution()
img_layer = statebuilder.ImageLayerConfig(
    client.info.image_source(),
)
base_sb = statebuilder.StateBuilder(
    [img_layer],
    client=client,
    resolution=viewer_resolution,
)
sbs.append(base_sb)
dfs.append(pd.DataFrame())


for compartment_label, group_data in (
    object_pred.query("l2_count > 10").reset_index().groupby("compartment_label")
):
    seg_layer = statebuilder.SegmentationLayerConfig(
        client.info.segmentation_source(),
        alpha_3d=0.3,
        name=compartment_label,
        color_column="color",
    )
    seg_layer.add_selection_map(selected_ids_column="object_id", color_column="color")

    dfs.append(group_data)
    sb = statebuilder.StateBuilder(
        [seg_layer],
        client=client,
        resolution=viewer_resolution,
    )
    sbs.append(sb)


bbox_mapper = statebuilder.BoundingBoxMapper(
    point_column_a="point_a", point_column_b="point_b"
)
annotation_layer = statebuilder.AnnotationLayerConfig(
    name="bounding_boxes",
    mapping_rules=bbox_mapper,
)
annotation_state = statebuilder.StateBuilder([annotation_layer], client=client)

dfs.append(bbox_df)
sbs.append(annotation_state)

sb = statebuilder.ChainedStateBuilder(sbs)
json_out = statebuilder.helpers.package_state(dfs, sb, client=client, return_as="json")
state_dict = json.loads(json_out)

statebuilder.StateBuilder(base_state=state_dict, client=client).render_state(
    return_as="html"
)

# %%
