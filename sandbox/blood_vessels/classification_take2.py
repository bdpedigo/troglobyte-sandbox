# %%
import time
from pathlib import Path

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from skops.io import load

from troglobyte.features import CAVEWrangler

# %%

client = CAVEclient("minnie65_phase3_v1")
client.materialize.version = 943

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=3)

# %%

target_df = (
    pd.read_csv("troglobyte-sandbox/data/blood_vessels/segments_per_branch_bbox.csv")
    .drop(columns="IDs")
    .set_index("root_id")
)

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

# %%
box_params["x_min"] = box_params["PointA_X"] * box_params["mip_res_X"]
box_params["y_min"] = box_params["PointA_Y"] * box_params["mip_res_Y"]
box_params["z_min"] = box_params["PointA_Z"] * box_params["mip_res_Z"]
box_params["x_max"] = box_params["PointB_X"] * box_params["mip_res_X"]
box_params["y_max"] = box_params["PointB_Y"] * box_params["mip_res_Y"]
box_params["z_max"] = box_params["PointB_Z"] * box_params["mip_res_Z"]

# %%

model = load(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/local_compartment_classifier_bd_boxes.skops"
)

out_path = Path("troglobyte-sandbox/results/vasculature")

pad_distance = 20_000

for i in range(len(box_params)):
    lower = box_params.iloc[i][["x_min", "y_min", "z_min"]].values
    upper = box_params.iloc[i][["x_max", "y_max", "z_max"]].values
    og_box = np.array([lower, upper])

    padded_box = og_box.copy()
    padded_box[0] -= np.array(pad_distance)
    padded_box[1] += np.array(pad_distance)

    box_name = str(og_box.astype(int).ravel()).strip("[]").replace(" ", "_")

    query_root_ids = (
        target_df.reset_index()
        .set_index(["X", "Y", "Z"])
        .loc[box_params.index[0]]["root_id"]
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

    features = wrangler.features_
    features.to_csv(out_path / f"vasculature_features_box={box_name}.csv")

print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
quit()
# %%
features = wrangler.features_.dropna().copy()

# %%
level2_ids = features.index.get_level_values("level2_id")
levelx = 6
level5_ids = client.chunkedgraph.get_roots(level2_ids, stop_layer=levelx)

features["levelx_id"] = level5_ids

features.reset_index().groupby("object_id")[["levelx_id", "level2_id"]].nunique()


# %%
(features["bd_boxes_predict_proba_dendrite"] > 0.9).mean()

# %%
grouping = "levelx_id"
object_posteriors = features.groupby(["levelx_id"])[
    [
        "bd_boxes_predict_proba_axon",
        "bd_boxes_predict_proba_dendrite",
        "bd_boxes_predict_proba_soma",
        "bd_boxes_predict_proba_glia",
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

import time

import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from nglui import statebuilder
from skops.io import load

from troglobyte.features import CAVEWrangler

sbs = []
dfs = []
img_layer, seg_layer = statebuilder.helpers.from_client(client)

box_mapper = statebuilder.BoundingBoxMapper(
    point_column_a="point_a", point_column_b="point_b"
)
box_layer = statebuilder.AnnotationLayerConfig(name="box", mapping_rules=box_mapper)
box_df = pd.DataFrame(
    {
        "point_a": [og_box[0] / np.array([8, 8, 40])],
        "point_b": [og_box[1] / np.array([8, 8, 40])],
    }
)
sbs.append(statebuilder.StateBuilder(layers=[img_layer, seg_layer, box_layer]))
dfs.append(box_df)

# sample_object_predictions = object_predictions.sample(20)

colors = sns.color_palette("pastel", n_colors=5).as_hex()
for i, (label, ids) in enumerate(object_predictions.groupby(object_predictions)):
    # ids = ids.sample(max(1, len(ids)))
    sub_df = ids.to_frame().reset_index()
    sub_df["color"] = colors[i]
    new_seg_layer = statebuilder.SegmentationLayerConfig(
        source=client.info.segmentation_source(),
        name=label,
        selected_ids_column=grouping,
        color_column="color",
        alpha_3d=0.3,
    )
    sbs.append(statebuilder.StateBuilder(layers=[new_seg_layer]))
    dfs.append(sub_df)

sb = statebuilder.ChainedStateBuilder(sbs)

sb.render_state(dfs, return_as="html")
