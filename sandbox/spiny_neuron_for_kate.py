# %%
import time

import numpy as np
import pyvista as pv
from caveclient import CAVEclient
from cloudvolume import Bbox
from tqdm.auto import tqdm

from neurovista import center_camera, to_mesh_polydata

root_id = 864691135489514810

client = CAVEclient("minnie65_public", version=1078)

cv = client.info.segmentation_cloudvolume()

# %%

nuc_loc = client.materialize.query_table(
    "nucleus_detection_v0", filter_equal_dict={"pt_root_id": root_id}
).iloc[0]["pt_position"] * np.array([4, 4, 40])

box_width = 50_000

bbox = np.array(
    [
        nuc_loc - box_width,
        nuc_loc + box_width,
    ]
)
bbox = bbox / np.array([8, 8, 40])
bbox = bbox.astype(int)
bbox = Bbox(bbox[0], bbox[1])

# %%

split = False
if split:
    currtime = time.time()
    node_ids = client.chunkedgraph.get_leaves(root_id, stop_layer=6)
    meshes = cv.mesh.get(
        node_ids,
        deduplicate_chunk_boundaries=False,
        allow_missing=True,
        bounding_box=bbox,
    )
    print(f"{time.time() - currtime:.3f} seconds elapsed to download mesh.")
else:
    currtime = time.time()
    mesh = cv.mesh.get(
        root_id,
        allow_missing=False,
        deduplicate_chunk_boundaries=False,
        remove_duplicate_vertices=False,
        bounding_box=bbox,
    )[root_id]
    print(f"{time.time() - currtime:.3f} seconds elapsed to download mesh.")

# %%


pv.set_jupyter_backend("client")
plotter = pv.Plotter()

currtime = time.time()
mesh_poly = to_mesh_polydata(mesh.vertices, mesh.faces)
# mesh_poly = mesh_poly.clean()
# mesh_poly = mesh_poly.decimate(0.90)
print(f"{time.time() - currtime:.3f} seconds elapsed to process mesh.")


# %%
currtime = time.time()
synapse_df = client.materialize.synapse_query(post_ids=root_id)
print(f"{time.time() - currtime:.3f} seconds elapsed to query synapses.")

# %%

synapse_positions = np.stack(synapse_df["post_pt_position"].values) * np.array(
    [4, 4, 40]
).astype(float)


point_poly = pv.PolyData(synapse_positions)
random_colors = np.random.randint(100, 255, size=(len(synapse_positions), 3)).astype(
    float
)

# can assign colors to be whatever you want here, i think rgb array is the way to go
point_poly["colors"] = random_colors


plotter = pv.Plotter(window_size=(3000, 1500))

animate = True
if animate:
    plotter.open_gif("neuron_with_synapses.gif", fps=30)

plotter.add_mesh(mesh_poly, color="cyan")

plotter.add_points(
    point_poly, scalars="colors", point_size=3, render_points_as_spheres=True, rgb=True
)

center_camera(plotter, nuc_loc, 1_000_000, up="-y", elevation=25)

if animate:
    azimuth_step_size = 1
    n_steps = 120
    for i in tqdm(range(n_steps)):
        plotter.camera.azimuth += azimuth_step_size
        plotter.write_frame()

    plotter.close()
else:
    plotter.show()
