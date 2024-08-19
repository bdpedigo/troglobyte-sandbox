# %%
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from caveclient import CAVEclient
from skops.io import load

from troglobyte.features import CAVEWrangler

# %%

client = CAVEclient("minnie65_public", version=1078)

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=3)

# %%
df = pd.read_csv(
    "troglobyte-sandbox/data/blood_vessels/segments_per_branch_2024-08-09.csv"
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

# %%
box_params["x_min"] = box_params["PointA_X"] * box_params["mip_res_X"]
box_params["y_min"] = box_params["PointA_Y"] * box_params["mip_res_Y"]
box_params["z_min"] = box_params["PointA_Z"] * box_params["mip_res_Z"]
box_params["x_max"] = box_params["PointB_X"] * box_params["mip_res_X"]
box_params["y_max"] = box_params["PointB_Y"] * box_params["mip_res_Y"]
box_params["z_max"] = box_params["PointB_Z"] * box_params["mip_res_Z"]

# %%
box_params.set_index(["x_min", "y_min", "z_min", "x_max", "y_max", "z_max"])[
    "BranchTypeName"
]

# %%

vessel_id = 864691137021018734

cv = client.info.segmentation_cloudvolume()
cv.cache.enabled = True

# %%
i = 7
box_info = box_params.iloc[i]
seg_res = np.array(client.chunkedgraph.segmentation_info["scales"][0]["resolution"])

# seg_res = np.array([4, 4, 40])

bounds_min_cg = (box_info[["x_min", "y_min", "z_min"]].values / seg_res).astype(int)
bounds_max_cg = (box_info[["x_max", "y_max", "z_max"]].values / seg_res).astype(int)

bounds = np.array([bounds_min_cg, bounds_max_cg])

sub_target_df = target_df[target_df["box_id"] == i]


from cloudvolume import Bbox

bounding_box = Bbox(bounds_min_cg, bounds_max_cg)

# cv.mesh.get(sub_target_df.index[:100], dedupli)

# cv.mesh.get(
#     sub_target_df.index[:10],
#     bounding_box=bounding_box,
#     deduplicate_chunk_boundaries=False,
#     remove_duplicate_vertices=False,
#     allow_missing=False,
# )


# %%

vessel_mesh = cv.mesh.get(
    vessel_id,
    bounding_box=bounding_box,
    deduplicate_chunk_boundaries=False,
    remove_duplicate_vertices=False,
    allow_missing=True,
)[vessel_id]


# %%
# low_res_path = "precomputed://https://rhoana.rc.fas.harvard.edu/ng/EM_lowres/mouse/bv"

# from cloudvolume import CloudVolume

# low_res_cv = CloudVolume(low_res_path)

# lower = box_info[["PointA_X", "PointA_Y", "PointA_Z"]]
# upper = box_info[["PointB_X", "PointB_Y", "PointB_Z"]]
# bbox_low_res = Bbox(lower, upper)

# img = low_res_cv.download(
#     bbox_low_res, coord_resolution=box_info[["mip_res_X", "mip_res_Y", "mip_res_Z"]]
# )

# img = img.squeeze()

# img.shape

# %%

# pad the image with one pixel of zeros on all sides

# img = np.pad(img, 1)

# #%%
# from skimage.segmentation import find_boundaries

# boundaries = find_boundaries(img, connectivity=1)

# # %%

# import pyvista as pv

# coords = np.nonzero(boundaries)

# coords = np.array(coords).T

# coords = coords * box_info[["mip_res_X", "mip_res_Y", "mip_res_Z"]].values.astype(int)

# coords = coords * np.array([8, 8, 40])

# coords = coords.astype(float)

# poly = pv.PolyData(coords)

# # poly = poly.reconstruct_surface()

# plotter = pv.Plotter()

# plotter.add_mesh(poly, color="red")

# plotter.show()


# %%

import pyvista as pv

from neurovista import to_mesh_polydata

pv.set_jupyter_backend("client")


vessel_mesh_poly = to_mesh_polydata(vessel_mesh.vertices, vessel_mesh.faces)

bounds_nm = bounds * seg_res


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
target_df

# %%
from nglui import statebuilder

client = CAVEclient("minnie65_public", version=1078)

img_layer, seg_layer = statebuilder.helpers.from_client(client)

sb = statebuilder.StateBuilder(
    layers=[img_layer, seg_layer],
    view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001},
)

sb.render_state(return_as="html", target_site="mainline")

# %%
from nglui import statebuilder

sbs = []
dfs = []
img_layer, seg_layer = statebuilder.helpers.from_client(client)

box_mapper = statebuilder.BoundingBoxMapper(
    point_column_a="point_a", point_column_b="point_b"
)
box_layer = statebuilder.AnnotationLayerConfig(name="box", mapping_rules=box_mapper)

rows = []
for id, box_info in box_params.iterrows():
    bounds_min_cg = (box_info[["x_min", "y_min", "z_min"]].values / seg_res).astype(int)
    bounds_max_cg = (box_info[["x_max", "y_max", "z_max"]].values / seg_res).astype(int)
    rows.append(
        {
            "point_a": bounds_min_cg,
            "point_b": bounds_max_cg,
        }
    )

box_df = pd.DataFrame(rows)
sbs.append(
    statebuilder.StateBuilder(
        layers=[img_layer, seg_layer, box_layer],
        view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001},
    )
)
dfs.append(box_df)

seg_layer = statebuilder.SegmentationLayerConfig(
    source="precomputed://https://rhoana.rc.fas.harvard.edu/ng/EM_lowres/mouse/bv",
    fixed_ids=[1],
    name="bv_low_res",
)
sbs.append(
    statebuilder.StateBuilder(
        layers=[seg_layer]  # view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001}
    )
)
dfs.append(pd.DataFrame())

sb = statebuilder.ChainedStateBuilder(sbs)

sb.render_state(dfs, return_as="html")  # target_site="mainline")


# %%

model = load(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/local_compartment_classifier_bd_boxes.skops"
)

out_path = Path("troglobyte-sandbox/results/vasculature") / "2024-08-09"

# pad_distance = 20_000
pad_distance = 30_000

for i in range(12, len(box_params)):
    lower = box_params.iloc[i][["x_min", "y_min", "z_min"]].values
    upper = box_params.iloc[i][["x_max", "y_max", "z_max"]].values
    og_box = np.array([lower, upper])

    padded_box = og_box.copy()
    padded_box[0] -= np.array(pad_distance)
    padded_box[1] += np.array(pad_distance)

    box_name = str(og_box.astype(int).ravel()).strip("[]").replace(" ", "_")

    query_root_ids = (
        target_df.reset_index()
        .set_index(["BranchX", "BranchY", "BranchZ"])
        .loc[box_params.index[i]]["root_id"]
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
    wrangler.stack_model_predict_proba("bd_boxes")

    features = wrangler.features_
    features.to_csv(out_path / f"vasculature_features_box={box_name}.csv")

    with open(out_path / f"wrangler_box={box_name}.pkl", "wb") as f:
        wrangler.client = None
        pickle.dump(wrangler, f)
        wrangler.client = client

print(f"{time.time() - currtime:.3f} seconds elapsed.")
