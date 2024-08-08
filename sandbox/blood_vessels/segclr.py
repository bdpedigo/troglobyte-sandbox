# %%
import gcsfs
import numpy as np

from connectomics.segclr import reader

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
type(embeddings_from_xyz)


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
future_id_map = time_maps["future_id_map"]


# %%
for id, past_ids in past_id_map.items():
    if len(past_ids) > 1:
        print(id, past_ids)
# %%

embedding_reader = reader.get_reader("microns_v343", PUBLIC_GCSFS)

embedding_reader[864691135293126156]

# %%

demo_test_id = 864691135293126156
print(client.chunkedgraph.get_root_timestamps(demo_test_id))
my_test_id = query_root_ids[2]
print(client.chunkedgraph.get_root_timestamps(my_test_id))

# %%
embedding_reader[demo_test_id]

# %%
from tqdm import tqdm

found_past_ids = []
embeddings = []
for root_id in tqdm(query_root_ids):
    past_ids = past_id_map[root_id]
    for past_id in past_ids:
        try:
            out = embedding_reader[past_id]
            embeddings.append(out)
            found_past_ids.append(past_id)
        except:
            continue

# %%

X = []
for embedding_for_root in embeddings:
    for xyz, embedding in embedding_for_root.items():
        X.append(embedding)
X = np.array(X)

# %%
from sklearn.decomposition import PCA

n_components = 10
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X)
X_pca = pd.DataFrame(X_pca, columns=[f"PC{i}" for i in range(n_components)])

# %%
import seaborn as sns

pg = sns.PairGrid(X_pca.sample(100_000), corner=True)

pg.map_lower(sns.scatterplot, s=15, alpha=0.3)

# %%
from umap import UMAP

n_neighbors = 20
min_dist = 0.7

umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)

X_umap = umap.fit_transform(X.sample(100_000))
X_umap = pd.DataFrame(X_umap, columns=["UMAP1", "UMAP2"])

# %%
pg = sns.PairGrid(X_umap, corner=True)

pg.map_lower(sns.scatterplot, s=15, alpha=0.3)


# %%
