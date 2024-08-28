# %%

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from caveclient import CAVEclient
from cloudvolume import Bbox
from skops.io import load

from minniemorpho.segclr import SegCLRQuery
from neurovista import to_mesh_polydata

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


for i in range(len(box_params)):
    box_info = box_params.iloc[i]
    box_name = box_info["BranchTypeName"]
    bounds_min_cg = (box_info[["x_min", "y_min", "z_min"]].values / seg_res).astype(int)
    bounds_max_cg = (box_info[["x_max", "y_max", "z_max"]].values / seg_res).astype(int)
    bounds = np.array([bounds_min_cg, bounds_max_cg])
    bounds_nm = bounds * seg_res

    sub_target_df = target_df[target_df["box_id"] == i]

    currtime = time.time()

    query = SegCLRQuery(client, verbose=True, n_jobs=-1, continue_on_error=True)
    query.set_query_ids(sub_target_df.index)
    query.set_query_bounds(bounds_nm)
    query.map_to_version()
    query.get_embeddings()
    query.map_to_level2()

    print(f"{time.time() - currtime:.3f} seconds elapsed.")

    features = query.features_
    mapping = query.level2_mapping_

    feature_cols = np.arange(64)
    predictions = model.predict(features[feature_cols].values)
    predictions = pd.Series(
        predictions, index=features.index, name="pred_label"
    ).to_frame()

    joined_features = features.join(mapping)
    joined_features = joined_features.join(predictions)

    joined_features.to_csv(out_path / f"{box_name}_segclr_features.csv.gz")

    print()

# %%
joined_features = pd.read_csv(
    out_path / f"{box_name}_segclr_features.csv.gz", index_col=[0, 1, 2, 3, 4]
)

# %%
client.chunkedgraph.get_leaves(
    query.query_ids[0], stop_layer=2, bounds=query.bounds_seg.astype(int).T
)

# %%

sample_roots = predictions.index.get_level_values("current_id").unique()
sample_roots = pd.Series(sample_roots, name="current_id").sample(300)

root_prediction_counts = predictions.to_frame().groupby("current_id").value_counts()
root_prediction_probs = (
    root_prediction_counts / root_prediction_counts.groupby("current_id").sum()
)
root_predictions = pd.DataFrame(
    {"pred_label": predictions.groupby("current_id").mode().values.flatten()}
)

# %%

weights = 1 / predictions.value_counts()
weights = weights / weights.sum()

# %%
bounding_box = Bbox(bounds_min_cg, bounds_max_cg)


# %%

vessel_id = 864691137021018734

cv = client.info.segmentation_cloudvolume()
cv.cache.enabled = True


cv.mesh.get(
    sub_target_df.index[:10],
    bounding_box=bounding_box,
    deduplicate_chunk_boundaries=False,
    remove_duplicate_vertices=False,
    allow_missing=False,
)


# %%

vessel_mesh = cv.mesh.get(
    vessel_id,
    bounding_box=bounding_box,
    deduplicate_chunk_boundaries=False,
    remove_duplicate_vertices=False,
    allow_missing=True,
)[vessel_id]


# %%


pv.set_jupyter_backend("client")


vessel_mesh_poly = to_mesh_polydata(vessel_mesh.vertices, vessel_mesh.faces)


def bounds_to_pyvista(bounds: np.ndarray) -> list:
    assert bounds.shape == (2, 3)
    return [
        bounds[0, 0],
        bounds[1, 0],
        bounds[0, 1],
        bounds[1, 1],
        bounds[0, 2],
        bounds[1, 2],
    ]


bbox_pyvista = bounds_to_pyvista(bounds_nm)

bbox_mesh = pv.Box(bounds=bbox_pyvista)


plotter = pv.Plotter()

plotter.add_mesh(vessel_mesh_poly.extract_largest(), color="red")
plotter.add_mesh(bbox_mesh, color="black", style="wireframe")

plotter.show()


# %%

meshes = cv.mesh.get(
    sample_roots,
    bounding_box=bounding_box,
    deduplicate_chunk_boundaries=False,
    remove_duplicate_vertices=False,
    allow_missing=False,
)
# %%

import seaborn as sns

# available_ids = features.index.get_level_values("current_id").unique()
available_ids = list(meshes.keys())

plotter = pv.Plotter()

colors = sns.color_palette("tab10")
palette = dict(zip(model.classes_, colors))

for current_id in available_ids:
    sub_predictions = predictions.loc[current_id].to_frame().reset_index()

    plurality_label = sub_predictions["pred_label"].mode().values[0]

    mesh = meshes[current_id]

    mesh_poly = to_mesh_polydata(mesh.vertices, mesh.faces)

    plotter.add_mesh(mesh_poly, color=palette[plurality_label], smooth_shading=True)

plotter.add_mesh(vessel_mesh_poly, color="red", smooth_shading=True)
plotter.add_mesh(bbox_mesh, color="black", style="wireframe", smooth_shading=True)

plotter.show()

# %%

palette

# %%
colors
