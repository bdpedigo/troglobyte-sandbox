# %%
import numpy as np
import pyvista as pv
from caveclient import CAVEclient
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


def to_mesh_polydata(
    nodes: np.ndarray,
    faces: np.ndarray,
):
    points = nodes.astype(float)

    faces = np.hstack([np.full((len(faces), 1), 3), faces])

    poly = pv.PolyData(points, faces=faces)

    return poly


client = CAVEclient("minnie65_public")
client.materialize.version = 1078

# %%
lumen_segments = [
    377858200361292074,
    377862598407805983,
    378421150314723072,
    378984100268135823,
    522910137985918230,
]

cv = client.info.segmentation_cloudvolume()

lumen_meshes = cv.mesh.get(lumen_segments, deduplicate_chunk_boundaries=False)

# %%


plotter = pv.Plotter()

bbox = np.array([[182528, 171776, 20160], [188416, 176384, 20544]])
bbox = bbox * np.array([4, 4, 40])


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


bbox_pyvista = bounds_to_pyvista(bbox)
bbox_mesh = pv.Box(bounds=bbox_pyvista)
plotter.add_mesh(bbox_mesh, color="black", style="wireframe")

points = []
polydatas = []
for seg_id, mesh in lumen_meshes.items():
    mesh_polydata = to_mesh_polydata(mesh.vertices, mesh.faces)
    polydatas.append(mesh_polydata)

lumen_polydata = pv.MultiBlock(polydatas)
lumen_polydata = lumen_polydata.extract_geometry()
lumen_polydata = lumen_polydata.clean()
lumen_polydata = lumen_polydata.decimate(0.90)
lumen_polydata = lumen_polydata.clip_box(bbox_pyvista, invert=False)
lumen_polydata = lumen_polydata.extract_largest()

plotter.add_mesh(lumen_polydata, color="red", opacity=0.5)
plotter.show()

points = np.array(lumen_polydata.points)

# %%

points.shape

neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1)
neighbors.fit(points)

# %%
sample_ids = [
    720575940386963165,
    864691135160805076,
    450606253341896986,
    738590338898595577,
    522910137985918512,
    738590338898419128,
    520587969428016532,
    864691133508212994,
    520658338172217744,
    450606253341961279,
    450606253341887112,
    522839769241812326,
    450588661155966791,
    449462761249092682,
    450588661155959162,
    450606253341899202,
    450606253341904777,
    522910137985927866,
    450588661155924440,
    522839769241814630,
]

sample_meshes = cv.mesh.get(sample_ids, deduplicate_chunk_boundaries=False)

polys = []
for i, mesh in tqdm(enumerate(sample_meshes.values()), total=len(sample_meshes)):
    mesh_polydata = to_mesh_polydata(mesh.vertices, mesh.faces)
    # mesh_polydata = mesh_polydata.clip_surface(bbox_mesh, invert=True)
    # mesh_polydata = mesh_polydata.clean()
    # if not mesh_polydata.is_all_triangles:
    # mesh_polydata = mesh_polydata.triangulate()
    # mesh_polydata = mesh_polydata.decimate(0.8)
    color = np.random.rand(3)
    color += 0.3
    color /= color.max()
    mesh_polydata["color"] = np.tile(color, (mesh_polydata.n_points, 1))

    dists, _ = neighbors.kneighbors(mesh_polydata.points, return_distance=True)
    mesh_polydata["lumen_min_dist"] = dists

    polys.append(mesh_polydata)

# %%
n_points_total = 0
for mesh_polydata in polys:
    n_points_total += mesh_polydata.n_points
print(n_points_total)

# %%
lumen_polydata.n_points

# %%
plotter = pv.Plotter()

plotter.open_gif("axon_meshes_test.gif", fps=30)

plotter.add_mesh(lumen_polydata, color="red", opacity=1)

plotter.add_mesh(bbox_mesh, color="black", style="wireframe", line_width=4)

joint_polydata = pv.MultiBlock(polys)
joint_polydata = joint_polydata.extract_geometry()

azimuth_step_size = 1
n_steps = 1
for thresh in tqdm(np.linspace(0, 10_000, 200)):
    thresh_polydata = joint_polydata.threshold(
        thresh, scalars="lumen_min_dist", invert=True
    )
    if thresh_polydata.n_points == 0:
        continue
    added_mesh = plotter.add_mesh(
        thresh_polydata, scalars="color", opacity=0.7, rgb=True, smooth_shading=True
    )
    for i in range(n_steps):
        plotter.camera.azimuth += azimuth_step_size
        plotter.write_frame()
    plotter.remove_actor(added_mesh)

plotter.close()

# %%
