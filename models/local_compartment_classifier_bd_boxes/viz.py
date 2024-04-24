import json
from typing import Optional, Union

import caveclient as cc
import cloudvolume
import numpy as np
import pandas as pd
import seaborn as sns
from nglui import statebuilder

from networkframe import NetworkFrame

# %%
y_pred = final_lda.predict(train_X_df)

# %%


# wrangler.query_level2_edges()
edges_by_object = wrangler.object_level2_edges_
all_nodes = wrangler.features_

class_to_int_map = {"soma": 1, "axon": 2, "dendrite": 3, "glia": 4, "unknown": 0}

networkframes_by_object = {}

classification_by_object = {}

for object_id in edges_by_object.index.get_level_values("root_id")[:200]:
    edges = edges_by_object.loc[object_id][["source", "target"]]
    nodes = all_nodes.loc[object_id].copy()
    temp_X = nodes.dropna()
    temp_X = temp_X.drop(columns=[col for col in temp_X.columns if "rep_coord" in col])
    if temp_X.empty:
        continue
    y_pred = pd.Series(final_lda.predict(temp_X), index=temp_X.index)
    classification_by_object[object_id] = (
        y_pred.value_counts().sort_values(ascending=False).index[0]
    )
    y_pred = y_pred.map(class_to_int_map).astype(int)
    nodes["predicted_compartment"] = 0
    nodes.loc[temp_X.index, "predicted_compartment"] = y_pred
    nodes.rename(
        columns={"rep_coord_x": "x", "rep_coord_y": "y", "rep_coord_z": "z"},
        inplace=True,
    )

    nf = NetworkFrame(nodes, edges)

    networkframes_by_object[object_id] = nf

classification_by_object = pd.Series(classification_by_object, name="compartment_label")

classification_by_object = classification_by_object.to_frame()

colors = sns.color_palette("tab10", n_colors=5).as_hex()
color_map = {k: colors[i] for i, k in enumerate(class_to_int_map.keys())}

classification_by_object["color"] = classification_by_object["compartment_label"].map(
    color_map
)
classification_by_object.index.name = "object_id"
classification_by_object.reset_index(inplace=True)


def write_networkframes_to_skeletons(
    networkframes: Union[NetworkFrame, dict[NetworkFrame]],
    client: cc.CAVEclient,
    attribute: Optional[str] = None,
    directory: str = "gs://allen-minnie-phase3/tempskel",
):
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
        vertices = networkframe.nodes[["x", "y", "z"]].values
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
    networkframes_by_object,
    client,
    attribute="predicted_compartment",
    directory="gs://allen-minnie-phase3/tempskel",
)

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

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
    "precomputed://gs://allen-minnie-phase3/tempskel"
)
skel_layer.add_selection_map(selected_ids_column="object_id")

base_sb = statebuilder.StateBuilder(
    [img_layer, seg_layer, skel_layer],
    client=client,
    resolution=viewer_resolution,
)
base_df = pd.DataFrame({"object_id": networkframes_by_object.keys()})

sbs.append(base_sb)
dfs.append(base_df)

sb = statebuilder.ChainedStateBuilder(sbs)
json_out = statebuilder.helpers.package_state(dfs, sb, client=client, return_as="json")
state_dict = json.loads(json_out)


shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    if (compartment == 0.0) {
        emitRGB(vec3(0.2, 0.2, 0.2));
    }
    if (compartment == 1.0) {
        emitRGB(vec3(0.9, 0.2, 0.2));
    }
    if (compartment == 2.0) {
        emitRGB(vec3(0.2, 0.9, 0.2));
    }
    if (compartment == 3.0) {
        emitRGB(vec3(0.2, 0.2, 0.9));
    }
    if (compartment == 4.0) {
        emitRGB(vec3(0.9, 0.9, 0.2));
    }
}
"""
skel_rendering_kws = {
    "shader": shader,
    "mode2d": "lines_and_points",
    "mode3d": "lines",
    "lineWidth3d": 1,
}

state_dict["layers"][1]["skeletonRendering"] = skel_rendering_kws

statebuilder.StateBuilder(base_state=state_dict, client=client).render_state(
    return_as="html"
)

# %%

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


for compartment_label, group_data in classification_by_object.groupby(
    "compartment_label"
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
bboxes = [
    eval(x.replace("array", "np.array")) for x in label_df["bbox_4x4x40"].unique()
]
bbox_df = pd.DataFrame(bboxes, columns=["point_a", "point_b"])
dfs.append(bbox_df)
sbs.append(annotation_state)

sb = statebuilder.ChainedStateBuilder(sbs)
json_out = statebuilder.helpers.package_state(dfs, sb, client=client, return_as="json")
state_dict = json.loads(json_out)


shader = """
void main() {
    float compartment = vCustom2;
    vec4 uColor = segmentColor();
    if (compartment == 0.0) {
        emitRGB(vec3(0.2, 0.2, 0.2));
    }
    if (compartment == 1.0) {
        emitRGB(vec3(0.9, 0.2, 0.2));
    }
    if (compartment == 2.0) {
        emitRGB(vec3(0.2, 0.9, 0.2));
    }
    if (compartment == 3.0) {
        emitRGB(vec3(0.2, 0.2, 0.9));
    }
    if (compartment == 4.0) {
        emitRGB(vec3(0.9, 0.9, 0.2));
    }
}
"""
skel_rendering_kws = {
    "shader": shader,
    "mode2d": "lines_and_points",
    "mode3d": "lines",
    "lineWidth3d": 1,
}

state_dict["layers"][1]["skeletonRendering"] = skel_rendering_kws


statebuilder.StateBuilder(base_state=state_dict, client=client).render_state(
    return_as="html"
)

# %%
