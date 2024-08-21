# %%
from pathlib import Path

import gcsfs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from caveclient import CAVEclient
from fast_simplification import simplify_mesh
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from umap import UMAP

from connectomics.segclr import reader
from neurovista import to_mesh_polydata

# %%

PUBLIC_GCSFS = gcsfs.GCSFileSystem(token="anon")

# %%


client = CAVEclient("minnie65_phase3_v1")
client.materialize.version = 1078


# %%
df = pd.read_csv(
    "troglobyte-sandbox/data/blood_vessels/segments_per_branch_2024-08-19.csv"
)
target_df = df.set_index("IDs")
target_df.index.name = "root_id"

# %%
box_params = target_df.groupby(["BranchX", "BranchY", "BranchZ"])[
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
        "BranchTypeName",
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
box_params.set_index(["x_min", "y_min", "z_min", "x_max", "y_max", "z_max"])[
    "BranchTypeName"
]
# %%
box_params = box_params.reset_index().set_index("BranchTypeName")


# %%
pad_distance = 10_000

# i = 0
branch = "CVT_Layernan_branch75"
i = box_params.index.get_loc(branch)

lower = box_params.iloc[i][["x_min", "y_min", "z_min"]].values
upper = box_params.iloc[i][["x_max", "y_max", "z_max"]].values
og_box = np.array([lower, upper])

padded_box = og_box.copy()
padded_box[0] -= np.array(pad_distance)
padded_box[1] += np.array(pad_distance)

# box_name = str(og_box.astype(int).ravel()).strip("[]").replace(" ", "_")
box_name = branch

query_root_ids = (
    target_df.reset_index()
    .set_index("BranchTypeName")
    .loc[box_params.index[i]]["root_id"]
).values

# %%
timestamp_343 = client.materialize.get_timestamp(343)

time_maps = client.chunkedgraph.get_past_ids(
    query_root_ids, timestamp_past=timestamp_343
)
past_id_map = time_maps["past_id_map"]

forward_id_map = {}
for current_id, past_ids in past_id_map.items():
    for past_id in past_ids:
        forward_id_map[past_id] = current_id


# %%

padded_box_cg = (padded_box / np.array([8, 8, 40])).astype(int)


out_path = Path("troglobyte-sandbox/results/vasculature/segclr")
pull_embeddings = False
threshold = 10
if pull_embeddings or not (out_path / f"embedding_df_{box_name}.csv.gz").exists():

    def get_n_leaves_for_root(root_id):
        root_infos = []
        past_ids = past_id_map[root_id]
        for past_id in past_ids:
            n_leaves = len(
                client.chunkedgraph.get_leaves(past_id, bounds=padded_box_cg.T)
            )
            root_infos.append(
                {"root_id": root_id, "past_root_id": past_id, "n_level2_ids": n_leaves}
            )
        return root_infos

    with tqdm_joblib(desc="Getting n_leaves", total=len(query_root_ids)):
        root_infos = Parallel(n_jobs=-1)(
            delayed(get_n_leaves_for_root)(root_id) for root_id in query_root_ids
        )
    # collapse into one list
    root_infos = [item for sublist in root_infos for item in sublist]
    root_info = pd.DataFrame(root_infos)

    out_path = Path("troglobyte-sandbox/results/vasculature/segclr")

    root_info.to_csv(out_path / f"root_info_{box_name}.csv.gz")

    big_past_root_ids = (
        root_info.groupby("past_root_id")["n_level2_ids"]
        .sum()
        .to_frame()
        .query(f"n_level2_ids >= {threshold}")
        .index
    )

    embedding_reader = reader.get_reader("microns_v343", PUBLIC_GCSFS)

    def get_embeddings_for_past_id(past_id):
        root_id = forward_id_map[past_id]
        try:
            out = embedding_reader[past_id]
            new_out = {}
            for xyz, embedding_vector in out.items():
                new_out[(root_id, past_id, *xyz)] = embedding_vector
            return new_out
        except Exception:
            return {}

    with tqdm_joblib(desc="Getting embeddings", total=len(big_past_root_ids)):
        embeddings_dicts = Parallel(n_jobs=-1)(
            delayed(get_embeddings_for_past_id)(past_id)
            for past_id in big_past_root_ids
        )

    embeddings_dict = {}
    for d in embeddings_dicts:
        embeddings_dict.update(d)

    embedding_df = pd.DataFrame(embeddings_dict).T
    embedding_df.index.names = ["root_id", "past_id", "x", "y", "z"]

    embedding_df["x_nm"] = embedding_df.index.get_level_values("x") * 32
    embedding_df["y_nm"] = embedding_df.index.get_level_values("y") * 32
    embedding_df["z_nm"] = embedding_df.index.get_level_values("z") * 40
    mystery_offset = np.array([13824, 13824, 14816]) * np.array([8, 8, 40])
    embedding_df["x_nm"] += mystery_offset[0]
    embedding_df["y_nm"] += mystery_offset[1]
    embedding_df["z_nm"] += mystery_offset[2]

    xmin = padded_box[0, 0]
    xmax = padded_box[1, 0]
    ymin = padded_box[0, 1]
    ymax = padded_box[1, 1]
    zmin = padded_box[0, 2]
    zmax = padded_box[1, 2]

    embedding_df = embedding_df.query(
        "x_nm > @xmin and x_nm < @xmax and y_nm > @ymin and y_nm < @ymax and z_nm > @zmin and z_nm < @zmax"
    )
    embedding_df = embedding_df.droplevel(["x", "y", "z"])
    embedding_df = embedding_df.set_index(
        ["x_nm", "y_nm", "z_nm"], drop=False, append=True
    )
    embedding_df.to_csv(out_path / f"embedding_df_{box_name}.csv.gz")

else:
    embedding_df = pd.read_csv(
        out_path / f"embedding_df_{box_name}.csv.gz", index_col=[0, 1, 2, 3, 4]
    )

# xmin = padded_box[0, 0]
# xmax = padded_box[1, 0]
# ymin = padded_box[0, 1]
# ymax = padded_box[1, 1]
# zmin = padded_box[0, 2]
# zmax = padded_box[1, 2]

# embedding_df = embedding_df.query(
#     "x_nm > @xmin and x_nm < @xmax and y_nm > @ymin and y_nm < @ymax and z_nm > @zmin and z_nm < @zmax"
# )
# embedding_df[["x_nm", "y_nm", "z_nm"]] = embedding_df[["x_nm", "y_nm", "z_nm"]].astype(
#     int
# )


# %%

pv.set_jupyter_backend("client")

plotter = pv.Plotter()
points = embedding_df[["x_nm", "y_nm", "z_nm"]].values
point_cloud = pv.PolyData(points)
plotter.add_mesh(point_cloud, point_size=1)


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


og_box_poly = pv.Box(bounds=bounds_to_pyvista(og_box))
padded_box_poly = pv.Box(bounds=bounds_to_pyvista(padded_box))

plotter.add_mesh(og_box_poly, color="black", style="wireframe")
plotter.add_mesh(padded_box_poly, color="red", style="wireframe")

plotter.show()

# %%
print(embedding_df.shape)

# %%


n_neighbors = 20
min_dist = 0.3

umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)

X = embedding_df.drop(columns=["x_nm", "y_nm", "z_nm"]).values
umap_df = umap.fit_transform(X)
umap_df = pd.DataFrame(umap_df, columns=["UMAP1", "UMAP2"], index=embedding_df.index)

# %%

fig, ax = plt.subplots()
sns.scatterplot(
    data=umap_df, x="UMAP1", y="UMAP2", ax=ax, s=0.2, alpha=0.5, linewidth=0
)

# %%

cv = client.info.segmentation_cloudvolume()
cv.cache.enabled = True

# %%
umap_color_df = umap_df.copy()

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(quantile_range=(2, 98))
scaled_X = scaler.fit_transform(umap_color_df)
print(scaled_X.min(), scaled_X.max())

scaled_X /= np.abs(scaled_X).max() * 2
print(scaled_X.min(), scaled_X.max())

scaled_X += 0.5

print(scaled_X.min(), scaled_X.max())


umap_color_df = pd.DataFrame(
    scaled_X,
    columns=umap_color_df.columns,
    index=umap_color_df.index,
)


# sns.scatterplot(data=umap_color_df, x="UMAP1", y="UMAP2", s=0.2, alpha=0.5, linewidth=0)

# X = umap_color_df[["UMAP1", "UMAP2"]].values

# # rotate into 3D
# X = np.hstack([X, 0.5 * np.ones((len(X), 1))])

# umap_color_df["x"] = X[:, 0]
# umap_color_df["y"] = X[:, 1]
# umap_color_df["z"] = X[:, 2]


# umap_color_df["UMAP1"] = umap_color_df["UMAP1"] - umap_color_df["UMAP1"].min()
# umap_color_df["UMAP1"] = (
#     umap_color_df["UMAP1"] / umap_color_df["UMAP1"].max() * 0.8 + 0.1
# )
# umap_color_df["UMAP2"] = umap_color_df["UMAP2"] - umap_color_df["UMAP2"].min()
# umap_color_df["UMAP2"] = (
#     umap_color_df["UMAP2"] / umap_color_df["UMAP2"].max() * 0.8 + 0.1
# )
# umap_color_df["UMAP3"] = umap_color_df["UMAP3"] - umap_color_df["UMAP3"].min()
# umap_color_df["UMAP3"] = (
#     umap_color_df["UMAP3"] / umap_color_df["UMAP3"].max() * 0.8 + 0.1
# )

# umap_color_df["UMAP1"] = umap_color_df["UMAP1"] * 100
# umap_color_df["UMAP2"] = umap_color_df["UMAP2"] * 255 - 128
# umap_color_df["UMAP3"] = umap_color_df["UMAP3"] * 255 - 128

# from colormath.color_conversions import convert_color
# from colormath.color_objects import sRGBColor
# from tqdm.auto import tqdm

# colors = []
# for i, (root_id, row) in enumerate(
#     tqdm(umap_color_df.iterrows(), total=len(umap_color_df))
# ):
#     # color = LabColor(
#     #     row["UMAP1"] * 100,
#     #     row["UMAP2"] * 255 - 128,
#     #     row["UMAP3"] * 255 - 128,
#     # )
#     # color = convert_color(color, sRGBColor)
#     # # color = color.get_value_tuple()
#     color = sRGBColor(
#         row["x"],
#         row["y"],
#         row["z"],
#     )
#     # color = HSLColor(
#     #     row["x"],
#     #     row["y"],
#     #     row["z"],
#     # )
#     color = convert_color(color, sRGBColor)
#     color = color.get_value_tuple()
#     colors.append(dict(zip(["r", "g", "b"], color)))

# colors_df = pd.DataFrame(colors, index=umap_color_df.index)

# umap_color_df = umap_color_df.join(colors_df)


# X = umap_color_df.sample(10000)
# plotter = pv.Plotter()
# plotter.add_mesh(
#     X[["x", "y", "z"]].values,
#     point_size=10,
#     scalars=X[["r", "g", "b"]].values,
#     rgb=True,
#     render_points_as_spheres=True,
# )
# plotter.show()

# %%

# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import AgglomerativeClustering
# n_clusters = 8
# # gmm = GaussianMixture(n_components=n_components, covariance_type="full", n_init=1)
# model = AgglomerativeClustering(n_clusters=n_clusters)
# labels = gmm.fit_predict(umap_color_df[["UMAP1", "UMAP2"]].values)
# umap_color_df["label"] = labels.astype(str)


from scipy.cluster.hierarchy import fcluster, linkage

n_colors = 8

method = "complete"
cluster_on = "embedding"
if cluster_on == "umap":
    X = umap_color_df[["UMAP1", "UMAP2"]].values
    metric = "euclidean"
elif cluster_on == "embedding":
    X = embedding_df.drop(columns=["x_nm", "y_nm", "z_nm"]).values
    metric = "euclidean"

subsample = 30_000
select_indices = np.random.choice(len(X), subsample, replace=False)
X_subsample = X[select_indices]

Z_subsample = linkage(X_subsample, method=method, metric=metric)
labels_subsample = fcluster(Z_subsample, n_colors, criterion="maxclust")

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_subsample, labels_subsample)

labels = knn.predict(X)

umap_color_df["label"] = labels.astype(str)
umap_color_df["label_prop"] = umap_color_df["label"].astype(float) / (n_colors - 1)

import seaborn as sns

cmap = sns.color_palette("husl", n_colors, as_cmap=True)

colors = cmap(umap_color_df["label_prop"])

umap_color_df["r"] = colors[:, 0]
umap_color_df["g"] = colors[:, 1]
umap_color_df["b"] = colors[:, 2]
umap_color_df["a"] = colors[:, 3]

import pyvista as pv

plotter = pv.Plotter()

# %%
fig, ax = plt.subplots()
sns.scatterplot(
    data=umap_color_df,
    x="UMAP1",
    y="UMAP2",
    ax=ax,
    s=0.2,
    alpha=0.5,
    linewidth=0,
    hue="label",
)


# %%
root_counts = embedding_df.groupby("root_id").size().sort_values(ascending=False)

# %%

from cloudvolume import Bbox
from sklearn.neighbors import NearestNeighbors

pull_meshes = False
if pull_meshes:
    test_roots = root_counts.index[100:150]

    box = Bbox(*padded_box_cg.tolist())

    cv_meshes = cv.mesh.get(
        test_roots, bounding_box=box, deduplicate_chunk_boundaries=False
    )

# %%

from tqdm.auto import tqdm

mesh_polys = {}
for test_root, mesh in tqdm(cv_meshes.items(), total=len(cv_meshes)):
    mesh_poly = to_mesh_polydata(mesh.vertices, mesh.faces)
    mesh_poly = (
        mesh_poly.clip_surface(padded_box_poly, invert=True)
        .extract_largest()
        .clean()
        .smooth(n_iter=100)
    )
    mesh_poly = simplify_mesh(mesh_poly, target_reduction=0.8, agg=8)

    sub_embedding_df = embedding_df.iloc[embedding_df.index.get_loc(test_root)]
    X = sub_embedding_df[["x_nm", "y_nm", "z_nm"]].values
    neighbors = NearestNeighbors(n_neighbors=1).fit(X)
    distances, indices = neighbors.kneighbors(mesh_poly.points)
    umap_colors = umap_color_df.loc[sub_embedding_df.index[indices.flatten()]]
    mesh_poly["colors"] = umap_colors[["r", "g", "b"]].values

    mesh_polys[test_root] = mesh_poly


# %%

plotter = pv.Plotter()

for test_root, mesh_poly in mesh_polys.items():
    plotter.add_mesh(mesh_poly, scalars="colors", rgba=True, smooth_shading=True)

plotter.add_mesh(og_box_poly, color="black", style="wireframe", line_width=5)
plotter.add_mesh(padded_box_poly, color="red", style="wireframe")

plotter.show()

# %%
