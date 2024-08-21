# %%
import time

import fast_simplification
import numpy as np
import pyvista as pv
from caveclient import CAVEclient
from cloudvolume import Mesh as CVMesh
from pymeshfix import MeshFix

from neurovista import bounds_to_box, to_mesh_polydata
from pyFM.mesh import TriMesh
from pyFM.signatures import mesh_HKS

pv.set_jupyter_backend("client")

client = CAVEclient("minnie65_public", version=1078)

root_ids = [864691135469080402]

nuc_info = client.materialize.query_table(
    "nucleus_detection_v0",
    filter_in_dict={"pt_root_id": root_ids},
    split_positions=True,
)
nuc_loc = nuc_info[["pt_position_x", "pt_position_y", "pt_position_z"]].values.squeeze()
nuc_loc = nuc_loc * np.array([4, 4, 40])

pad = 100_000
# pad = 30_000
box_min = nuc_loc - pad
box_max = nuc_loc + pad
bounds = np.array([box_min, box_max])

box = bounds_to_box(bounds)

cv = client.info.segmentation_cloudvolume()
cv.cache.enabled = True

# %%

object_meshes = cv.mesh.get(root_ids, deduplicate_chunk_boundaries=False)


# %%


def process_mesh(mesh: CVMesh, crop=False):
    mesh_poly = to_mesh_polydata(mesh.vertices, mesh.faces)
    if crop:
        mesh_poly = mesh_poly.clip_surface(box)
    mesh_poly = mesh_poly.clean().triangulate().smooth(n_iter=100).extract_largest()
    mesh_poly = fast_simplification.simplify_mesh(
        mesh_poly, target_reduction=0.8, agg=8
    ).extract_largest()
    mesh_fix = MeshFix(mesh_poly)
    mesh_fix.repair()
    mesh_poly = mesh_fix.mesh
    return mesh_poly


currtime = time.time()

crop = True
object_polydatas = {
    mesh_id: process_mesh(mesh, crop=crop) for mesh_id, mesh in object_meshes.items()
}

print(f"{time.time() - currtime:.3f} seconds elapsed to preprocess.")


polydata = object_polydatas[root_ids[0]]


# %%
plotter = pv.Plotter()
for polydata in object_polydatas.values():
    plotter.add_mesh(polydata)
plotter.add_mesh(box, color="red", style="wireframe", line_width=2)
plotter.show()

# %%
print(polydata)

# %%


def polydata_to_decomposed_trimesh(polydata: pv.PolyData, k=800):
    faces = polydata.faces.reshape(-1, 4)[:, 1:]
    points = polydata.points
    trimesh = TriMesh(points, faces)

    trimesh.process(k=k, skip_normals=True, intrinsic=False, robust=False, verbose=True)

    return trimesh


currtime = time.time()
object_trimeshes = {
    object_id: polydata_to_decomposed_trimesh(polydata)
    for object_id, polydata in object_polydatas.items()
}
print(f"{time.time() - currtime:.3f} seconds elapsed to eigendecompose.")


# %%


def compute_hks(
    trimesh,
    time_scales=None,
    n_components=16,
):
    if time_scales is not None:
        t_factor = time_scales
    else:
        t_factor = n_components

    hks = mesh_HKS(trimesh, t_factor)
    return hks


hks_by_object = {}

t_min = 1e4
t_max = 1e8  # had 1e12
n_components = 64
time_scales = np.geomspace(t_min, t_max, n_components)


currtime = time.time()

for object_id, polydata in object_trimeshes.items():
    if object_id in hks_by_object:
        continue
    # try:
    hks = compute_hks(polydata, time_scales=time_scales)
    hks_by_object[object_id] = hks
    # except Exception as e:
    #     print(e)
    #     print(f"Failed to compute HKS for {object_id}")
    #     continue
print(f"{time.time() - currtime:.3f} seconds elapsed to compute HKS.")

# %%

i = -1
# j = 20

plotter = pv.Plotter()

polydata = object_polydatas[object_id]
polydata["hks"] = np.log(hks[:, i])
plotter.add_mesh(polydata, scalars="hks", show_scalar_bar=False)
plotter.show()


# %%

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_context("talk")

X_df = pd.DataFrame(data=hks)
X_train_df = X_df.sample(20_000).copy()

X_train_df["index"] = X_train_df.index
X_train_df_long = X_train_df.melt(
    var_name="component", value_name="hks", id_vars="index"
)
X_train_df = X_train_df.drop(columns="index")
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.lineplot(
    data=X_train_df_long,
    x="component",
    y="hks",
    alpha=0.01,
    estimator=None,
    units="index",
)
ax.set(ylabel="HKS", xlabel="Time scale")
ax.set_yscale("log")

# %%


# sns.clustermap(
#     np.log(X_train_df.values).T, cmap="Reds", figsize=(10, 10), row_cluster=False
# )


# %%
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.neighbors import KNeighborsClassifier

method = "average"
metric = "euclidean"
n_neighbors = 5
n_clusters = 8
X = np.log(X_train_df.values)

currtime = time.time()

Z = linkage(X, method=method, metric=metric)
labels = fcluster(Z, n_clusters, criterion="maxclust")

husl = sns.color_palette("husl", n_clusters, as_cmap=True)
colors = husl(labels / n_clusters)


knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X, labels)

full_labels = knn.predict(np.log(X_df.values))
print(f"{time.time() - currtime:.3f} seconds elapsed to cluster.")

# %%

sns.clustermap(
    X.T,
    cmap="Reds",
    figsize=(10, 5),
    row_cluster=False,
    col_linkage=Z,
    col_colors=colors,
    xticklabels=False,
    yticklabels=False,
)


# %%
polydata["label"] = full_labels / n_clusters

cmap = sns.color_palette("tab20", n_clusters, as_cmap=True)

plotter = pv.Plotter()
plotter.add_mesh(polydata, scalars="label", cmap=cmap)
plotter.show()


# %%

n_samples = 20
uni_labels = np.unique(full_labels)
plotter = pv.Plotter(shape=(len(uni_labels), n_samples))
for j, label in enumerate(uni_labels):
    label_poly = polydata.extract_points(np.where(full_labels == label)[0])
    bodies = label_poly.split_bodies()
    for i, body in enumerate(bodies[:n_samples]):
        body.points = body.points - body.center
        plotter.subplot(j, i)
        plotter.add_mesh(body, color=cmap(label / n_clusters))
plotter.show()

# %%

polydata["hks"] = hks

pad = 30_000
# pad = 30_000
box_min = nuc_loc - pad
box_max = nuc_loc + pad
small_bounds = np.array([box_min, box_max])
small_box = bounds_to_box(small_bounds)

cropped_poly = polydata.clip_surface(small_box)
# cropped_poly.plot(scalars="hks")

cropped_hks = np.array(cropped_poly["hks"])

plotter = pv.Plotter()

i = 0
plotter.add_mesh(cropped_poly, scalars=np.log(cropped_hks[:, i]), show_scalar_bar=True)

plotter.show()

# %%

pv.set_jupyter_backend("trame")
plotter = pv.Plotter()

actors = []


def select_scale(i):
    for actor in actors:
        plotter.remove_actor(actor)
    i = np.floor(i).astype(int)
    polydata = object_polydatas[object_id]
    polydata["hks"] = np.log(hks[:, i])
    actor = plotter.add_mesh(polydata, scalars="hks", show_scalar_bar=False)
    actors.append(actor)


plotter.add_slider_widget(select_scale, (0.0, float(n_components - 1)), value=0)

plotter.show()

# %%

import pandas as pd

hks_df = pd.DataFrame(data=hks)


# %%

i = 0
n_rows = 4
plotter = pv.Plotter(shape=(n_rows, np.ceil(n_components / n_rows).astype(int)))
for object_id, hks in hks_by_object.items():
    for j in range(n_components):
        row_ind, col_ind = np.unravel_index(
            j, (n_rows, np.ceil(n_components / n_rows).astype(int))
        )
        plotter.subplot(row_ind, col_ind)
        polydata = object_polydatas[object_id]
        polydata["hks"] = np.log(hks[:, j])
        plotter.add_mesh(polydata, scalars="hks", show_scalar_bar=False)
# plotter.link_views()
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
