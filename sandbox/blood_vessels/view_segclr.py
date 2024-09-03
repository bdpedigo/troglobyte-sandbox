# %%

from pathlib import Path

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from skops.io import load

# %%

client = CAVEclient("minnie65_public", version=1078)

# %%
df = pd.read_csv(
    "troglobyte-sandbox/data/blood_vessels/segments_per_branch_2024-08-19.csv"
)
target_df = df.set_index("IDs")
target_df.index.name = "root_id"
target_df.groupby(["BranchX", "BranchY", "BranchZ"]).size().sort_values()

# %%
box_params = target_df.groupby(["BranchX", "BranchY", "BranchZ"])[
    [
        "NearestBranchX",
        "NearestBranchY",
        "NearestBranchZ",
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
box_params = box_params.sort_values(["BranchX", "BranchY", "BranchZ"])
box_params["box_id"] = np.arange(len(box_params))

box_params["NearestBranchX"] = box_params["NearestBranchX"].astype(int)
box_params["NearestBranchY"] = box_params["NearestBranchY"].astype(int)
box_params["NearestBranchZ"] = box_params["NearestBranchZ"].astype(int)

box_params = box_params.reset_index().set_index("box_id")

box_params

# %%
target_df["box_id"] = target_df["BranchTypeName"].map(
    box_params.reset_index().set_index("BranchTypeName")["box_id"]
)
box_params["x_min"] = box_params["PointA_X"] * box_params["mip_res_X"]
box_params["y_min"] = box_params["PointA_Y"] * box_params["mip_res_Y"]
box_params["z_min"] = box_params["PointA_Z"] * box_params["mip_res_Z"]
box_params["x_max"] = box_params["PointB_X"] * box_params["mip_res_X"]
box_params["y_max"] = box_params["PointB_Y"] * box_params["mip_res_Y"]
box_params["z_max"] = box_params["PointB_Z"] * box_params["mip_res_Z"]
box_params["volume"] = (
    (box_params["x_max"] - box_params["x_min"])
    * (box_params["y_max"] - box_params["y_min"])
    * (box_params["z_max"] - box_params["z_min"])
)

# %%

out_path = Path("troglobyte-sandbox/data/blood_vessels/segclr/2024-08-19")
seg_res = np.array(client.chunkedgraph.segmentation_info["scales"][0]["resolution"])

model_path = Path("minniemorpho/models/segclr_logreg_bdp.skops")
model = load(model_path)

i = 1
# for i in range(len(box_params)):
box_info = box_params.iloc[i]
box_name = box_info["BranchTypeName"]
bounds_min_cg = (box_info[["x_min", "y_min", "z_min"]].values / seg_res).astype(int)
bounds_max_cg = (box_info[["x_max", "y_max", "z_max"]].values / seg_res).astype(int)
bounds = np.array([bounds_min_cg, bounds_max_cg])
bounds_nm = bounds * seg_res

# %%
joined_features = pd.read_csv(
    out_path / f"{box_name}_segclr_features.csv.gz", index_col=[0, 1, 2, 3, 4]
)

# %%

seg_df = joined_features[["level2_id", "distance_to_level2_node", "pred_label"]].copy()

# %%
timestamp = client.timestamp

level2_ids = seg_df["level2_id"].unique()
max_level = 6
min_level = 4
for level in range(min_level, max_level + 1):
    print(level)
    level_ids = client.chunkedgraph.get_roots(
        level2_ids, stop_layer=level, timestamp=timestamp
    )
    level_map = dict(zip(level2_ids, level_ids))
    seg_df[f"level{level}_id"] = seg_df["level2_id"].map(level_map)

# %%
level_names = [f"level{level}_id" for level in range(min_level, max_level + 1)]
level_names += ["current_id"]


mixed_level_df = []
for level_name in level_names:
    groupby = seg_df.groupby(level_name)
    label_counts = groupby["pred_label"].value_counts()
    label_props = label_counts / label_counts.groupby(level_name).transform("sum")
    label_counts = label_counts.unstack().fillna(0)
    label_props = label_props.unstack().fillna(0)
    label_props = label_props.rename(columns=lambda x: f"{x}_prop")

    new_df = label_props

    new_df.index.name = "node_id"
    new_df.index = new_df.index.astype("Int64")
    if level_name == "object_id":
        level_name = "root_id"
    new_df["level"] = level_name
    new_df["label"] = new_df[[f"{cl}_prop" for cl in model.classes_]].idxmax(axis=1)
    new_df["label"] = new_df["label"].str.replace("_prop", "")
    mixed_level_df.append(new_df.reset_index())

mixed_level_df = pd.concat(mixed_level_df, ignore_index=True).reset_index(drop=True)
# %%

import numpy as np
import seaborn as sns
from nglui import statebuilder
from nglui.segmentprops import SegmentProperties

colors = sns.color_palette("tab10").as_hex()
palette = dict(
    zip(["dendrite", "axon", "glia", "perivascular", "soma", "thick/myelin"], colors)
)


seg_df = mixed_level_df.copy()
# seg_df = seg_df.sample(20000)

n_randoms = 1
for i in range(n_randoms):
    seg_df[f"random_{i}"] = np.random.uniform(0, 1, size=len(seg_df))


prop_urls = {}
for label, label_seg_df in seg_df.groupby("label"):
    seg_prop = SegmentProperties.from_dataframe(
        label_seg_df.reset_index(),
        id_col="node_id",
        label_col="node_id",
        tag_value_cols=["level"],
        number_cols=[f"random_{i}" for i in range(n_randoms)],
        # + ["lumen_min_dist"]
        # + [f"{cl}_prop" for cl in model.classes_],
    )

    prop_id = client.state.upload_property_json(seg_prop.to_dict())
    prop_url = client.state.build_neuroglancer_url(
        prop_id, format_properties=True, target_site="mainline"
    )
    prop_urls[label] = prop_url

# import json

# cf = CloudFiles("gs://allen-minnie-phase3/ben-segprops/vasculature")
# cf.put_json("info.json", json.dumps(seg_prop.to_dict()))

# json.dump(
#     seg_prop.to_dict(), open("troglobyte-sandbox/data/blood_vessels/segprops.json", "w")
# )

client = CAVEclient("minnie65_public")
client.materialize.version = 1078

img = statebuilder.ImageLayerConfig(
    source=client.info.image_source(),
)

n_samples = 100
base_level = 6
seg_layers = []
for label, prop_url in prop_urls.items():
    label_seg_df = seg_df[seg_df["label"] == label]
    sample_ids = label_seg_df.query(f"level == 'level{base_level}_id'")["node_id"]
    if len(sample_ids) > n_samples:
        sample_ids = sample_ids.sample(n_samples)
    seg = statebuilder.SegmentationLayerConfig(
        name=label,
        source=client.info.segmentation_source(),
        segment_properties=prop_url,
        active=False,
        fixed_ids=sample_ids,
        fixed_id_colors=[palette[label]] * len(sample_ids),
        # fixed_ids=lumen_segments,
        skeleton_source="precomputed://middleauth+https://minnie.microns-daf.com/skeletoncache/api/v1/minnie65_phase3_v1/precomputed/skeleton",
        # view_kws={"visible": False},
    )
    seg_layers.append(seg)


box_map = statebuilder.BoundingBoxMapper(
    point_column_a="point_column_a", point_column_b="point_column_b"
)
ann = statebuilder.AnnotationLayerConfig(
    name="box", mapping_rules=box_map, data_resolution=[1, 1, 1]
)
box_df = pd.DataFrame(
    [
        {
            "point_column_a": [box_info["x_min"], box_info["y_min"], box_info["z_min"]],
            "point_column_b": [box_info["x_max"], box_info["y_max"], box_info["z_max"]],
        }
    ]
)


sb = statebuilder.StateBuilder(
    layers=[img] + seg_layers + [ann],
    target_site="mainline",
    # view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001},
    client=client,
)

state_dict = sb.render_state(data=box_df, return_as="dict")

for layer in state_dict["layers"]:
    if layer["type"] == "segmentation":
        layer["visible"] = False

state_dict["layout"] = "3d"

sb = statebuilder.StateBuilder(
    base_state=state_dict,
    target_site="mainline",
    client=client,
)
sb.render_state(return_as="html")

# %%
