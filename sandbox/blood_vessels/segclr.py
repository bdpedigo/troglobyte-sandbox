# %%
import gcsfs
import numpy as np

from connectomics.segclr import reader

# from connectomics.segclr.classification import model_handler

# had to install tensorflow-probability, tf-keras

# model, configs = model_handler.load_model(
#     "gs://iarpa_microns/minnie/minnie65/embedding_classification/models/subcompartment_10um_BERT_SNGP_20220819"
# )

# %%

PUBLIC_GCSFS = gcsfs.GCSFileSystem(token="anon")

test_id_from_prefix = dict(
    h01=1014493630,
    microns=864691135293126156,
)

for data_key in sorted(reader.DATA_URL_FROM_KEY_BYTEWIDTH64):
    print(data_key)
    embedding_reader = reader.get_reader(data_key, PUBLIC_GCSFS)
    print("embedding_reader:", embedding_reader)
    test_id = None
    for id_prefix in test_id_from_prefix:
        if data_key.startswith(id_prefix):
            test_id = test_id_from_prefix[id_prefix]
    print("test_id", test_id)
    embeddings_from_xyz = embedding_reader[test_id]
    print(f"Test {data_key} segment ID:", test_id)
    print("#embedding rows:", len(embeddings_from_xyz))
    print("example xyz->embedding tuple:", next(iter(embeddings_from_xyz.items())))
    print()

# %%


import pandas as pd
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")
client.materialize.version = 1078


# %%
df = pd.read_csv(
    "troglobyte-sandbox/data/blood_vessels/segments_per_branch_2024-07-11.csv"
)
target_df = df.set_index("IDs")
target_df.index.name = "root_id"
# target_df = target_df.sample(2)

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


def get_n_leaves_for_root(root_id):
    root_infos = []
    past_ids = past_id_map[root_id]
    for past_id in past_ids:
        n_leaves = len(client.chunkedgraph.get_leaves(past_id, bounds=padded_box_cg.T))
        root_infos.append(
            {"root_id": root_id, "past_root_id": past_id, "n_leaves": n_leaves}
        )
    return root_infos


from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

with tqdm_joblib(desc="Getting n_leaves", total=len(query_root_ids)):
    root_infos = Parallel(n_jobs=-1)(
        delayed(get_n_leaves_for_root)(root_id) for root_id in query_root_ids
    )
# collapse into one list
root_infos = [item for sublist in root_infos for item in sublist]


# %%
import seaborn as sns

root_info = pd.DataFrame(root_infos)
sns.histplot(data=root_info.groupby("root_id").sum(), x="n_leaves", log_scale=True)


# %%
for current_id, past_ids in past_id_map.items():
    if 864691133041929973 in past_ids:
        print(current_id)
        break

past_id_map[864691133041929973]

# %%
client.chunkedgraph.get_past_ids([864691136195393868], timestamp_past=timestamp_343)

# %%
client.chunkedgraph.get_past_ids([864691134885087643], timestamp_past=timestamp_343)

# %%

embedding_reader = reader.get_reader("microns_v343", PUBLIC_GCSFS)

# %%
big_past_root_ids = (
    root_info.groupby("past_root_id")["n_leaves"]
    .sum()
    .to_frame()
    .query("n_leaves > 20")
    .index
)


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


# found_past_ids = []
# embeddings_dict = {}
# for past_id in tqdm(big_past_root_ids[:200]):
#     root_id = forward_id_map[past_id]
#     try:
#         out = embedding_reader[past_id]
#         new_out = {}
#         for xyz, embedding_vector in out.items():
#             new_out[(root_id, past_id, *xyz)] = embedding_vector
#         embeddings_dict.update(new_out)
#         found_past_ids.append(past_id)
#     except Exception:
#         continue

with tqdm_joblib(desc="Getting embeddings", total=len(big_past_root_ids)):
    embeddings_dicts = Parallel(n_jobs=-1)(
        delayed(get_embeddings_for_past_id)(past_id) for past_id in big_past_root_ids
    )

# %%
embeddings_dict = {}
for d in embeddings_dicts:
    embeddings_dict.update(d)

# %%
past_id_info = []
for past_id in big_past_root_ids[:]:
    shard = embedding_reader._sharder(past_id)
    past_id_info.append({"past_id": past_id, "shard": shard})

past_id_info = pd.DataFrame(past_id_info)
sns.histplot(past_id_info["shard"].value_counts(), discrete=True)

# %%
embedding_df = pd.DataFrame(embeddings_dict).T
embedding_df.index.names = ["root_id", "past_id", "x", "y", "z"]

# %%
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

# %%
import pyvista as pv

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
cv = client.info.segmentation_cloudvolume()
bounds = cv.bounds

cg_bounds = bounds.to_list()

# %%

test_id = (embedding_df.groupby("past_id").size() - 1001).abs().idxmin()
# test_id = 864691135981371228

test_mesh = cv.mesh.get(test_id)[test_id]


def to_mesh_polydata(
    nodes: np.ndarray,
    faces: np.ndarray,
):
    points = nodes.astype(float)

    faces = np.hstack([np.full((len(faces), 1), 3), faces])

    poly = pv.PolyData(points, faces=faces)

    return poly


test_mesh = to_mesh_polydata(test_mesh.vertices, test_mesh.faces)
test_mesh.plot()

# %%
points = embedding_df.droplevel("root_id").loc[test_id, ["x_nm", "y_nm", "z_nm"]].values

# %%

pv.set_jupyter_backend("client")
volume_bounds = np.array([cg_bounds[:3], cg_bounds[3:]])
volume_bounds = volume_bounds * np.array([8, 8, 40])

box = pv.Box(
    [
        volume_bounds[0, 0],
        volume_bounds[1, 0],
        volume_bounds[0, 1],
        volume_bounds[1, 1],
        volume_bounds[0, 2],
        volume_bounds[1, 2],
    ]
)

plotter = pv.Plotter()
# plotter.add_mesh(box, color="black", style="wireframe")
# plotter.add_mesh(point_cloud, point_size=0.1)

# plotter.add_mesh(points, color="red", point_size=2, style="wireframe", opacity=0.5)


# points_moved = points + volume_bounds[0] * np.array([0.5, 0.5, 1])
# plotter.add_mesh(points_moved, color="blue", point_size=2)

points_moved2 = points + np.array([13824, 13824, 14816]) * np.array([8, 8, 40])
plotter.add_mesh(points_moved2, color="green", point_size=10)


plotter.add_mesh(test_mesh, color="red", opacity=0.5)

plotter.show()

# %%


# %%


# xyzs = sorted(embeddings)
# embedding_array = np.array([embeddings[_] for _ in xyzs])


# %%
from sklearn.decomposition import PCA

X = embedding_df.values
n_components = 10
pca = PCA(n_components=n_components, whiten=True)

X_pca = pca.fit_transform(X)
X_pca = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(n_components)])


# %%
import seaborn as sns

X_df = pd.DataFrame(X)
pg = sns.PairGrid(X_df.iloc[:, :8], corner=True)

pg.map_lower(sns.scatterplot, s=15, alpha=0.3)

# %%
pg = sns.PairGrid(X_pca, corner=True)

pg.map_lower(sns.scatterplot, s=15, alpha=0.3)


# %%

# n_neighbors = 20
# min_dist = 0.3

# umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)

# X_umap = umap.fit_transform(X)
# X_umap = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])

# %%
# pg = sns.PairGrid(X_umap, corner=True)

# pg.map_lower(sns.scatterplot, linewidth=0)


# %%
