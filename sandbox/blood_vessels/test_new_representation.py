# %%
import cloudvolume as cv
from caveclient import CAVEclient

from neurovista import to_mesh_polydata

client = CAVEclient("minnie65_public", version=1078)

lumen_segments = [
    377858200361292074,
    377862598407805983,
    378421150314723072,
    378984100268135823,
    522910137985918230,
]


cv = client.info.segmentation_cloudvolume()
cv.cache.enabled = True
lumen_meshes = cv.mesh.get(lumen_segments, deduplicate_chunk_boundaries=False)

lumen_polydatas = {
    mesh_id: to_mesh_polydata(mesh.vertices, mesh.faces)
    for mesh_id, mesh in lumen_meshes.items()
}

# %%
import fast_simplification
import pyvista as pv

pv.set_jupyter_backend("client")
# lumen_polydata = pv.MultiBlock(list(lumen_polydatas.values())).combine().triangulate()
lumen_polydata = (
    pv.MultiBlock(list(lumen_polydatas.values())).combine().extract_surface().clean()
)
lumen_polydata = lumen_polydata.smooth(n_iter=100)
lumen_polydata = fast_simplification.simplify_mesh(
    lumen_polydata, target_reduction=0.9, agg=9
)
lumen_polydata["curvature"] = lumen_polydata.curvature(curv_type="mean")
lumen_polydata.plot(
    scalars="curvature",
    clim=[-0.005, 0.005],
)

# %%
# level 6 ids, so fairly local
axons = [
    377858200361289476,
    377858200361289533,
    377858200361289558,
    377858200361289559,
    377858200361289565,
    377858200361289569,
    377858200361289600,
    377858200361289644,
    377858200361289728,
    377858200361289753,
    377858200361289859,
    377858200361289952,
    377858200361290006,
    377858200361290322,
    377858200361291490,
    377858200361291565,
    449462761249079035,
    449462761249113938,
    449462761249090046,
]
dendrites = [
    377858200361289504,
    377858200361289523,
    377858200361289554,
    377858200361289625,
    377858200361289638,
    377858200361289717,
    377858200361289747,
    377858200361289766,
    377858200361289773,
    377858200361289963,
    377858200361290005,
    377858200361291972,
    377858200361292329,
    738590338898311412,
    449462761249099603,
    449462761249134618,
    449462761249154917,
]
myelinated_axons = [
    450588661155958147,
]
lumens = [
    377858200361292074,
    377862598407805983,
    378421150314723072,
    378984100268135823,
    522910137985918230,
]

object_ids = axons + dendrites + myelinated_axons + lumens

object_meshes = cv.mesh.get(object_ids, deduplicate_chunk_boundaries=False)


# %%
object_polydatas = {
    mesh_id: to_mesh_polydata(mesh.vertices, mesh.faces)
    for mesh_id, mesh in object_meshes.items()
}

new_object_polydatas = {}
for object_id, polydata in object_polydatas.items():
    polydata = polydata.clean().smooth(n_iter=100, inplace=True).triangulate()
    polydata = fast_simplification.simplify_mesh(
        polydata, target_reduction=0.8, agg=7
    ).extract_largest()
    polydata.compute_implicit_distance(lumen_polydata, inplace=True)
    new_object_polydatas[object_id] = polydata

object_polydatas = new_object_polydatas

# %%
plotter = pv.Plotter()
clim = [0, 10_000]
for polydata in object_polydatas.values():
    plotter.add_mesh(polydata, scalars="implicit_distance", clim=clim)

# plotter.add_mesh(lumen_polydata, color="red")

plotter.show()

# %%
import numpy as np

from pyFM.mesh import TriMesh
from pyFM.signatures import mesh_HKS


def compute_hks(polydata, n_components=16, time_scales=None, k=50):
    faces = polydata.faces.reshape(-1, 4)[:, 1:]
    points = polydata.points
    trimesh = TriMesh(points, faces)

    trimesh.process(k=k, skip_normals=True, intrinsic=False, robust=False, verbose=True)

    if time_scales is not None:
        t_factor = time_scales
    else:
        t_factor = n_components

    hks = mesh_HKS(trimesh, t_factor)
    return hks


hks_by_object = {}

t_min = 1e-1
t_max = 1e10
n_components = 64
time_scales = np.geomspace(t_min, t_max, n_components)

# %%

for object_id, polydata in object_polydatas.items():
    if object_id in hks_by_object:
        continue
    try:
        hks = compute_hks(polydata, time_scales=time_scales)
        hks_by_object[object_id] = hks
    except:
        continue

# %%
i = 0
plotter = pv.Plotter()
for object_id, hks in hks_by_object.items():
    polydata = object_polydatas[object_id]
    n_points = polydata.n_points
    polydata["hks"] = np.log(hks[:, i]) / np.log(n_points**2)
    plotter.add_mesh(polydata, scalars="hks")
plotter.show()

# %%

col = 5
n_objects = len(hks_by_object)
n_rows = 3
n_cols = n_objects // n_rows + 1
plotter = pv.Plotter(shape=(n_rows, n_cols))
for i, (object_id, hks) in enumerate(hks_by_object.items()):
    row_ind, col_ind = np.unravel_index(i, (n_rows, n_cols))
    plotter.subplot(row_ind, col_ind)
    polydata = object_polydatas[object_id]
    n_points = polydata.n_points
    polydata["hks"] = hks[:, col] / n_points
    clim = hks[:, col].min(), hks[:, col].max()
    plotter.add_mesh(polydata, scalars="hks", clim=clim)
# plotter.link_views()
plotter.show()

# %%

import matplotlib.pyplot as plt
import seaborn as sns

col = 30
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for i, (object_id, hks) in enumerate(hks_by_object.items()):
    sns.histplot(np.log(hks[:, col]), bins=100, kde=True, label=object_id)


# %%

polydata["hks0"] = hks[:, 0]

# %%
x = 0
polydata["hksx"] = hks[:, x]
polydata.plot(scalars="hksx")

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

all_hks = np.concatenate([hks for hks in hks_by_object.values()], axis=0)

pca = PCA(n_components=2)

hks_pca = pca.fit_transform(all_hks)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sns.scatterplot(
    x=hks_pca[:, 0],
    y=hks_pca[:, 1],
    # hue=hks_pca[:, 2],
    ax=ax,
)


# %%

plotter = pv.Plotter()
clim = [0, 10_000]
for polydata in object_polydatas.values():
    edges = polydata.extract_all_edges()
    plotter.add_mesh(edges, style="wireframe", color="black")

# plotter.add_mesh(lumen_polydata, color="red")

plotter.show()

# %%
line_info = edges.lines
# every third element, starting with index 1, is the source
# every third element, starting with index 2, is the target
sources = line_info[1::3]
targets = line_info[2::3]

# %%
# construct a laplacian from the edges
import numpy as np
from scipy.sparse import csr_matrix

n_points = polydata.n_points
data = np.ones(len(sources))

adjacency = csr_matrix(
    (data, (sources, targets)),
    shape=(n_points, n_points),
)

from scipy.sparse.csgraph import laplacian

laplacian_op = laplacian(adjacency, normed=False, symmetrized=True)

# %%
from scipy.sparse.linalg import eigsh

# lu = splu(laplacian_op)
# op_inv = LinearOperator(
#     matvec=lu.solve, shape=laplacian_op.shape, dtype=laplacian_op.dtype
# )

eigsh(laplacian_op, k=10, which="SM")
