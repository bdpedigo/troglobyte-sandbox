# %%

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
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

test = True

for i in range(len(box_params)):
    box_info = box_params.iloc[i]
    box_name = box_info["BranchTypeName"]
    bounds_min_cg = (box_info[["x_min", "y_min", "z_min"]].values / seg_res).astype(int)
    bounds_max_cg = (box_info[["x_max", "y_max", "z_max"]].values / seg_res).astype(int)
    bounds = np.array([bounds_min_cg, bounds_max_cg])
    bounds_nm = bounds * seg_res

    sub_target_df = target_df[target_df["box_id"] == i]
    query_ids = sub_target_df.index
    if test:
        query_ids = query_ids[:1000]

    currtime = time.time()

    query = SegCLRQuery(
        client, verbose=True, n_jobs=-1, continue_on_error=True, version=943
    )
    query.set_query_ids(query_ids)

    query.set_query_bounds(bounds_nm)
    query.map_to_version()
    query.get_embeddings()
    query.map_to_level2()

    print(f"{time.time() - currtime:.3f} seconds elapsed.")

    feature_cols = np.arange(64)
    features = query.features_[feature_cols]
    mapping = query.level2_mapping_

    predictions = model.predict(features[feature_cols].values)
    predictions = pd.Series(
        predictions, index=features.index, name="pred_label"
    ).to_frame()
    posteriors = model.predict_proba(features[feature_cols].values)
    posteriors = pd.DataFrame(
        posteriors, index=features.index, columns=model.classes_
    ).add_prefix("posterior_")

    predictions = predictions.join(posteriors)

    joined_features = features.join(mapping)
    joined_features = joined_features.join(predictions)

    if not test:
        joined_features.to_csv(out_path / f"{box_name}_segclr_features.csv.gz")

    print()

    if test:
        break

# %%

sample_roots = predictions.index.get_level_values("current_id").unique()
sample_roots = pd.Series(sample_roots, name="current_id").sample(
    min(200, len(sample_roots))
)

# %%
bounding_box = Bbox(bounds_min_cg, bounds_max_cg)


# %%

vessel_id = 864691137021018734

cv = client.info.segmentation_cloudvolume()
cv.cache.enabled = True


# %%
pv.set_jupyter_backend("client")


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


vessel_mesh = cv.mesh.get(
    vessel_id,
    bounding_box=bounding_box,
    deduplicate_chunk_boundaries=False,
    remove_duplicate_vertices=False,
    allow_missing=True,
)
if vessel_id not in vessel_mesh:
    pass
else:
    vessel_mesh_poly = to_mesh_polydata(vessel_mesh.vertices, vessel_mesh.faces)

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


# available_ids = features.index.get_level_values("current_id").unique()
available_ids = list(meshes.keys())

plotter = pv.Plotter()

colors = sns.color_palette("tab10")
palette = dict(zip(model.classes_, colors))

for current_id in available_ids:
    sub_predictions = predictions.loc[current_id].reset_index()

    plurality_label = sub_predictions["pred_label"].mode().values[0]

    mesh = meshes[current_id]

    mesh_poly = to_mesh_polydata(mesh.vertices, mesh.faces)

    plotter.add_mesh(mesh_poly, color=palette[plurality_label], smooth_shading=True)

if vessel_id in vessel_mesh:
    plotter.add_mesh(vessel_mesh_poly, color="red", smooth_shading=True)

plotter.add_mesh(
    bbox_mesh, color="black", style="wireframe", smooth_shading=True, line_width=3
)

plotter.show()

# %%
