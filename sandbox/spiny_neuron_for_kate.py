# %%
import pickle
import time
from typing import Literal, Union

import numpy as np
import pyvista as pv
import seaborn as sns
from caveclient import CAVEclient
from fast_simplification import simplify_mesh
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

UP_MAP = {
    "x": (1, 0, 0),
    "y": (0, 1, 0),
    "z": (0, 0, 1),
    "-x": (-1, 0, 0),
    "-y": (0, -1, 0),
    "-z": (0, 0, -1),
}


def center_camera(
    plotter: pv.Plotter,
    center: np.ndarray,
    distance: float,
    up: Literal["x", "y", "z", "-x", "-y", "-z"] = "-y",
    elevation: Union[float, int] = 25,
):
    plotter.camera_position = "zx"
    plotter.camera.focal_point = center
    plotter.camera.position = center + np.array([0, 0, distance])
    plotter.camera.up = UP_MAP[up]
    plotter.camera.elevation = elevation


def to_mesh_polydata(
    nodes: np.ndarray,
    faces: np.ndarray,
):
    points = nodes.astype(float)

    faces = np.hstack([np.full((len(faces), 1), 3), faces])

    poly = pv.PolyData(points, faces=faces)

    return poly


root_id = 864691135489514810

client = CAVEclient("minnie65_public", version=1078)

cv = client.info.segmentation_cloudvolume()

# %%

nuc_loc = client.materialize.query_table(
    "nucleus_detection_v0", filter_equal_dict={"pt_root_id": root_id}
).iloc[0]["pt_position"] * np.array([4, 4, 40])

# %%

bbox = None
split = False
pull_mesh = False

if pull_mesh:
    currtime = time.time()
    mesh = cv.mesh.get(
        root_id,
        allow_missing=False,
        deduplicate_chunk_boundaries=False,
        remove_duplicate_vertices=False,
        bounding_box=bbox,
    )[root_id]
    print(f"{time.time() - currtime:.3f} seconds elapsed to download mesh.")

    currtime = time.time()
    mesh_poly = to_mesh_polydata(mesh.vertices, mesh.faces)

    mesh_poly = simplify_mesh(mesh_poly, target_reduction=0.9, agg=9, verbose=True)
    print(f"{time.time() - currtime:.3f} seconds elapsed to load and simplify mesh.")

    with open("mesh_poly.pkl", "wb") as f:
        pickle.dump(mesh_poly, f)
else:
    with open("mesh_poly.pkl", "rb") as f:
        mesh_poly = pickle.load(f)


# %%
currtime = time.time()
synapse_df = client.materialize.synapse_query(post_ids=root_id)
print(f"{time.time() - currtime:.3f} seconds elapsed to query synapses.")

# %%

box_width = 30_000
bbox = np.array(
    [
        nuc_loc - box_width,
        nuc_loc + box_width,
    ]
)
bbox = bbox / np.array([4, 4, 40])
x_min, y_min, z_min = bbox[0]
x_max, y_max, z_max = bbox[1]
synapse_df["pre_pt_position_x"] = synapse_df["pre_pt_position"].apply(lambda x: x[0])
synapse_df["pre_pt_position_y"] = synapse_df["pre_pt_position"].apply(lambda x: x[1])
synapse_df["pre_pt_position_z"] = synapse_df["pre_pt_position"].apply(lambda x: x[2])

# TODO can assign colors to be whatever you want here

colors = sns.husl_palette(n_colors=synapse_df.shape[0])
np.random.shuffle(colors)

synapse_df["color"] = colors

n_samples = 40

sub_synapses = (
    synapse_df.query(
        "@x_min < pre_pt_position_x < @x_max and @y_min < pre_pt_position_y < @y_max and @z_min < pre_pt_position_z < @z_max"
    )
    .sample(n_samples)
    .drop_duplicates("pre_pt_root_id", keep="first")
)

layer = 6
node_ids = []
sample_meshes = {}
for i, row in sub_synapses.iterrows():
    supervoxel_id = row["pre_pt_supervoxel_id"]
    node_id = client.chunkedgraph.get_roots(supervoxel_id, stop_layer=layer)[0]
    mesh = cv.mesh.get(node_id, allow_missing=True, deduplicate_chunk_boundaries=False)[
        node_id
    ]
    sample_meshes[i] = mesh

# %%

sample_mesh_polys = {}
for synapse_idx, sample_mesh in sample_meshes.items():
    synapse_position = sub_synapses.loc[synapse_idx]["pre_pt_position"] * np.array(
        [4, 4, 40]
    )
    sample_mesh_poly = to_mesh_polydata(sample_mesh.vertices, sample_mesh.faces)
    sample_mesh_poly = simplify_mesh(sample_mesh_poly, target_reduction=0.9, agg=9)
    dists = pairwise_distances(sample_mesh_poly.points, synapse_position.reshape(1, -1))
    sample_mesh_poly["distances"] = dists
    sample_mesh_polys[synapse_idx] = sample_mesh_poly

# %%

synapse_positions = np.stack(synapse_df["pre_pt_position"].values) * np.array(
    [4, 4, 40]
).astype(float)


point_poly = pv.PolyData(synapse_positions)


point_poly["colors"] = colors  # np.stack(synapse_df["color"].values)

pv.set_jupyter_backend("trame")

animate = True
fps = 20
pbr = False
window_size = (3000, 1500)
window_size = (1500, 750)
if animate:
    plotter = pv.Plotter(window_size=window_size)
    plotter.open_gif("neuron_with_synapses.gif", fps=fps)
else:
    plotter = pv.Plotter()

# plotter.background_color = "black"

plotter.add_mesh(mesh_poly, color="cyan", pbr=pbr)

plotter.add_points(
    point_poly,
    scalars="colors",
    # point_size=0.1,
    point_size=5,
    render_points_as_spheres=True,
    rgb=True,
    # style="points_gaussian",
)

center_camera(plotter, nuc_loc, 1_000_000, up="-y", elevation=25)

azimuth_step_size = 1
first_circle_time = 2
zoom_time = 5
zoom_factor = 1.03
second_circle_time = 6
threshold_factor = 200
if animate:
    for i in tqdm(range(first_circle_time * fps)):
        plotter.camera.azimuth += azimuth_step_size
        plotter.write_frame()

    for i in tqdm(range(zoom_time * fps)):
        plotter.camera.azimuth += azimuth_step_size
        plotter.zoom_camera(zoom_factor)
        plotter.write_frame()

    for i in tqdm(range(second_circle_time * fps)):
        plotter.camera.azimuth += azimuth_step_size

        actors = []
        for synapse_idx, sample_mesh_poly in sample_mesh_polys.items():
            thresh_mesh = sample_mesh_poly.threshold(
                i * threshold_factor, scalars="distances", invert=True
            )
            if thresh_mesh.n_points > 0:
                actor = plotter.add_mesh(
                    thresh_mesh,
                    color=synapse_df.loc[synapse_idx]["color"],
                    pbr=pbr,
                )
                actors.append(actor)

        plotter.write_frame()

        for actor in actors:
            plotter.remove_actor(actor)

    plotter.close()
else:
    plotter.show()
