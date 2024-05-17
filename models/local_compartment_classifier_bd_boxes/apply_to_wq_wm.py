# %%
from pathlib import Path

import caveclient as cc
import numpy as np
from skops.io import load

from troglobyte.features import CAVEWrangler

out_path = Path("./troglobyte-sandbox/models/")

model_name = "local_compartment_classifier_bd_boxes"


# %%
# model = load(out_path / model_name / f"{model_name}_no_syn.skops")
model = load(out_path / model_name / f"{model_name}.skops")

# %%
client = cc.CAVEclient("minnie65_public")
client.materialize.version = 943


# %%

# w-q box in white matter
start = np.array([263697, 279652, 21026])
end = np.array([266513, 282468, 21314])

# w-q box new box in white matter
start = np.array([304910, 252614, 20496])
end = np.array([329598, 281450, 23096])
# start = np.array([140252, 231547, 21012])
# end = np.array([143068, 234363, 21300])

import pandas as pd

object_ids = pd.read_csv("wq-ids.csv").columns.str.strip(" ").astype(int)

diff = end - start
box = np.array([start, end])

client.chunkedgraph.is_valid_nodes(object_ids.to_list())

# %%
from nglui import statebuilder

statebuilder.make_neuron_neuroglancer_link(client, [864691134836990722])

# %%
for oid in object_ids[:100]:
    if len(client.chunkedgraph.get_leaves(oid, stop_layer=2)) > 30:
        print(oid)
        break

# %%
# wrangler.query_objects_from_box(
#     box, source_resolution=np.array([4, 4, 40]), size_threshold=200
# )

# define a larger bbox for looking at features
# new_start = start - diff
# new_end = end + diff
# new_box = np.array([new_start, new_end])
# wrangler.set_query_box(new_box, source_resolution=np.array([4, 4, 40]))

#
wrangler = CAVEWrangler(client=client, verbose=10, n_jobs=-1)

# wrangler.n_jobs = 1

wrangler.set_objects(object_ids)
wrangler.set_query_box(box, source_resolution=np.array([4, 4, 40]))
wrangler.query_level2_ids()
wrangler.query_level2_edges()
wrangler.query_level2_shape_features()
wrangler.query_level2_synapse_features()
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5
)

# %%
box = np.array([[152455, 164799], [126307, 140725], [20496, 23096]])
# box = None
other_box = np.stack([start, end])
other_box = other_box / np.array([2, 2, 1])
other_box = other_box.astype(int).T
# client.chunkedgraph.level2_chunk_graph(864691134836990722, bounds=other_box)
client.chunkedgraph.level2_chunk_graph(864691136773648494, bounds=box)

# %%
wrangler.features_
# %%
X_df = wrangler.features_

X_df = X_df.drop(columns=[col for col in X_df.columns if "rep_coord" in col])
X_df = X_df.drop(columns=[col for col in X_df.columns if "pca_unwrapped" in col])
X_df = X_df.dropna()

others_df = pd.read_csv(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/features.csv",
    index_col=[0, 1],
)
others_df = others_df.dropna()
others_df = others_df[others_df.columns.intersection(X_df.columns)]


y = len(X_df) * ["wm_axon"] + len(others_df) * ["not_wm_axon"]

X_df = pd.concat([X_df, others_df])

# %%

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

model = Pipeline(
    [
        ("scaler", QuantileTransformer(output_distribution="normal")),
        ("lda", LinearDiscriminantAnalysis()),
    ]
)

model.fit(X_df, y)

y_pred = model.predict(X_df)

print(classification_report(y, y_pred))

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=400, max_depth=4)

rf.fit(X_df, y)
y_pred = rf.predict(X_df)

print(classification_report(y, y_pred))


# %%
X_trans_df = model.transform(X_df)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.histplot(x=X_trans_df[:, 0], hue=y, element="step", ax=ax)

# %%
oos_wrangler = CAVEWrangler(client=client, verbose=10, n_jobs=-1)

new_start = start + np.array([40000, 0, -3000])
new_end = end + np.array([40000, 0, -3000])
new_box = np.array([new_start, new_end])
oos_wrangler.set_query_box(new_box, source_resolution=np.array([4, 4, 40]))
oos_wrangler.query_objects_from_box(mip=5, sample=100)

# %%
oos_wrangler.query_level2_ids()

# %%
oos_wrangler.query_level2_edges()
oos_wrangler.query_level2_shape_features()
oos_wrangler.query_level2_synapse_features()
oos_wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5
)

# %%

new_X = oos_wrangler.features_

new_X = new_X[X_df.columns]
new_X = new_X.dropna()

# #%%
# new_y = model.predict(new_X)

# #%%
# new_y = pd.Series(new_y, index=new_X.index, name="compartment_label").to_frame()


y_pred = model.predict(new_X)

import pandas as pd

y_pred = pd.Series(y_pred, index=new_X.index, name="compartment_label").to_frame()

oos_wrangler.register_model(model, "compartment_label")

# %%

oos_wrangler.models_

# %%
oos_wrangler.stack_model_predict_proba("compartment_label", dropna=True)

# %%
oos_wrangler.query_level2_networks()
networks = oos_wrangler.object_level2_networks_


# %%

from typing import Optional, Union

import cloudvolume

from networkframe import NetworkFrame


def write_networkframes_to_skeletons(
    networkframes: Union[NetworkFrame, dict[NetworkFrame]],
    client: cc.CAVEclient,
    attribute: Optional[str] = None,
    spatial_columns: Optional[list[str]] = None,
    directory: str = "gs://allen-minnie-phase3/tempskel",
):
    if spatial_columns is None:
        spatial_columns = ["x", "y", "z"]
    # register an info file and set up CloudVolume
    base_info = client.chunkedgraph.segmentation_info
    base_info["skeletons"] = "skeleton"
    info = base_info.copy()

    cv = cloudvolume.CloudVolume(
        f"precomputed://{directory}",
        mip=0,
        info=info,
        compress=False,
    )
    cv.commit_info()

    sk_info = cv.skeleton.meta.default_info()
    sk_info["vertex_attributes"] = [
        {"id": "radius", "data_type": "float32", "num_components": 1},
        {"id": "vertex_types", "data_type": "float32", "num_components": 1},
    ]
    cv.skeleton.meta.info = sk_info
    cv.skeleton.meta.commit_info()

    sks = []
    if isinstance(networkframes, NetworkFrame):
        networkframes = {0: networkframes}

    for name, networkframe in networkframes.items():
        # extract vertex information
        vertices = networkframe.nodes[spatial_columns].values
        edges_unmapped = networkframe.edges[["source", "target"]].values
        edges = networkframe.nodes.index.get_indexer_for(
            edges_unmapped.flatten()
        ).reshape(edges_unmapped.shape)

        vertex_types = networkframe.nodes[attribute].values.astype(np.float32)

        radius = np.ones(len(vertices), dtype=np.float32)

        sk_cv = cloudvolume.Skeleton(
            vertices,
            edges,
            radius,
            None,
            segid=name,
            extra_attributes=sk_info["vertex_attributes"],
            space="physical",
        )
        sk_cv.vertex_types = vertex_types

        sks.append(sk_cv)

    cv.skeleton.upload(sks)


write_networkframes_to_skeletons(
    networks,
    client,
    attribute="compartment_label_predict_proba_1",
    spatial_columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"],
)

# %%
import json

networkframes_by_object = networks

sbs = []
dfs = []
viewer_resolution = client.info.viewer_resolution()
img_layer = statebuilder.ImageLayerConfig(
    client.info.image_source(),
)
seg_layer = statebuilder.SegmentationLayerConfig(
    client.info.segmentation_source(), alpha_3d=0.3
)
seg_layer.add_selection_map(selected_ids_column="object_id")

skel_layer = statebuilder.SegmentationLayerConfig(
    "precomputed://gs://allen-minnie-phase3/tempskel", name="skel"
)
skel_layer.add_selection_map(selected_ids_column="object_id")

base_sb = statebuilder.StateBuilder(
    [img_layer, seg_layer, skel_layer],
    client=client,
    resolution=viewer_resolution,
)
base_df = pd.DataFrame({"object_id": list(networkframes_by_object.keys())[:]})
sbs.append(base_sb)
dfs.append(base_df)


bbox_mapper = statebuilder.BoundingBoxMapper(
    point_column_a="point_a", point_column_b="point_b"
)
annotation_layer = statebuilder.AnnotationLayerConfig(
    name="bounding_boxes",
    mapping_rules=bbox_mapper,
)
annotation_state = statebuilder.StateBuilder([annotation_layer], client=client)
bbox_df = pd.DataFrame(
    {"point_a": [new_start], "point_b": [new_end]},
)

dfs.append(bbox_df)
sbs.append(annotation_state)


sb = statebuilder.ChainedStateBuilder(sbs)
json_out = statebuilder.helpers.package_state(dfs, sb, client=client, return_as="json")
state_dict = json.loads(json_out)


shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    emitRGB(0.5*uColor + vec4(0.5, 0.5, 0.5, 0.5)*compartment);
}
"""
skel_rendering_kws = {
    "shader": shader,
    "mode2d": "lines_and_points",
    "mode3d": "lines",
    "lineWidth3d": 1,
}
state_dict["layers"][-2]["skeletonRendering"] = skel_rendering_kws
statebuilder.StateBuilder(base_state=state_dict, client=client).render_state(
    return_as="html"
)

# %%
import seaborn as sns

class_to_int_map = {"soma": 1, "axon": 2, "dendrite": 3, "glia": 4, "unknown": 0}
class_to_int_map = {"wm_axon": 2, "not_wm_axon": 3}
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

object_pred["mid_level_id"] = None
mid_level = 5

for object, object_data in tqdm(object_pred.groupby("object_id")):
    l2_ids = object_data.index.get_level_values("level2_id")
    mid_level_ids = client.chunkedgraph.get_roots(l2_ids, stop_layer=mid_level)
    object_pred.loc[object_data.index, "mid_level_id"] = mid_level_ids


# %%
certain_y_pred = certain_y_pred.reset_index()
certain_y_pred["color"] = certain_y_pred["compartment_label"].map(color_map)

# %%
import json

from nglui import statebuilder

new_start = start + np.array([40000, 0, -3000])
new_end = end + np.array([40000, 0, -3000])
bbox_df = pd.DataFrame(
    {"point_a": [start, new_start], "point_b": [end, new_end]},
)


sbs = []
dfs = []
layers = []
viewer_resolution = client.info.viewer_resolution()
img_layer = statebuilder.ImageLayerConfig(
    client.info.image_source(),
    name="img",
)
base_sb = statebuilder.StateBuilder(
    [img_layer],
    client=client,
    resolution=viewer_resolution,
)
sbs.append(base_sb)
dfs.append(pd.DataFrame())

# seg_layer = statebuilder.SegmentationLayerConfig(
#     client.info.segmentation_source(),
#     alpha_3d=0.3,
#     name="seg",
# )
# # seg_layer.add_selection_map(selected_ids_column="object_id")
# seg_sb = statebuilder.StateBuilder(
#     [img_layer, seg_layer],
#     client=client,
#     resolution=viewer_resolution,
# )
# dfs.append(pd.DataFrame({"object_id": wrangler.object_ids_}))
# sbs.append(seg_sb)
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
        [img_layer, seg_layer],
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

from tqdm.auto import tqdm

features = oos_wrangler.features_.dropna().copy()
features["mid_level_id"] = None
mid_level = 2

for object, object_data in tqdm(features.groupby("object_id")):
    l2_ids = object_data.index.get_level_values("level2_id")
    mid_level_ids = client.chunkedgraph.get_roots(l2_ids, stop_layer=mid_level)
    features.loc[object_data.index, "mid_level_id"] = mid_level_ids

# %%
threshold = 0.5

features["compartment_label"] = (
    features["compartment_label_predict_proba_1"] > threshold
).map({True: "wm_axon", False: "not_wm_axon"})

features["color"] = features["compartment_label"].map(color_map)

mid_level_features = features.reset_index()

mid_level_features = (
    mid_level_features.groupby("mid_level_id")["compartment_label_predict_proba_1"]
    .mean()
    .to_frame()
)
mid_level_features["compartment_label"] = (
    mid_level_features["compartment_label_predict_proba_1"] > threshold
).map({True: "wm_axon", False: "not_wm_axon"})
mid_level_features["color"] = mid_level_features["compartment_label"].map(color_map)
mid_level_features["object_id"] = mid_level_features.index.map(
    features.reset_index().groupby("mid_level_id")["object_id"].first()
)
# %%

sbs = []
dfs = []
layers = []
viewer_resolution = client.info.viewer_resolution()
img_layer = statebuilder.ImageLayerConfig(
    client.info.image_source(),
    name="img",
)
base_sb = statebuilder.StateBuilder(
    [img_layer],
    client=client,
    resolution=viewer_resolution,
)
sbs.append(base_sb)
dfs.append(pd.DataFrame())


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

select_objects = (
    features.index.get_level_values("object_id").unique().to_series().sample(n=5)
)
select_mid_features = mid_level_features.query("object_id in @select_objects")

for compartment_label, group_data in select_mid_features.groupby("compartment_label"):
    seg_layer = statebuilder.SegmentationLayerConfig(
        client.info.segmentation_source(),
        alpha_3d=0.3,
        name=compartment_label,
        color_column="color",
    )
    seg_layer.add_selection_map(
        selected_ids_column="mid_level_id", color_column="color"
    )
    dfs.append(group_data.reset_index())
    sb = statebuilder.StateBuilder(
        [img_layer, seg_layer],
        client=client,
        resolution=viewer_resolution,
    )
    sbs.append(sb)


sb = statebuilder.ChainedStateBuilder(sbs)
json_out = statebuilder.helpers.package_state(dfs, sb, client=client, return_as="json")
state_dict = json.loads(json_out)

statebuilder.StateBuilder(base_state=state_dict, client=client).render_state(
    return_as="html"
)

# %%
