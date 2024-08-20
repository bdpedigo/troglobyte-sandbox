# %%
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from caveclient import CAVEclient
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from tqdm.autonotebook import tqdm

# client = CAVEclient("minnie65_phase3_v1")
client = CAVEclient("minnie65_public")
client.materialize.version = 1078

# original labels from Bethanny
feature_df = pd.read_csv(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/features_new.csv",
    index_col=[0, 1],
)
label_df = pd.read_csv(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/labels.csv",
    index_col=[0, 1],
)
label_df = label_df.rename(columns=lambda x: x.replace(".1", ""))
labels = label_df.droplevel("bbox_id")["simple_label"]
labels = labels.dropna()

labeled = feature_df.index.get_level_values("object_id").intersection(labels.index)
feature_df = feature_df.loc[labeled].dropna()
labels = labels.loc[feature_df.index.get_level_values("object_id").unique()]
labels.name = "label"

labels_by_level2 = (
    feature_df.index.get_level_values("object_id")
    .map(labels)
    .to_series(index=feature_df.index)
)

feature_df["label"] = labels_by_level2

# drop the previous soma labels
feature_df = feature_df.query("label != 'soma'")

# %%

data_path = Path("troglobyte-sandbox/data/blood_vessels")

new_features = []

new_label_map = {
    "perivascular": [
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4890727126401024"
    ],
    "soma": [
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6074591643435008",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6362053351571456",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5716916632027136",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5990023284391936",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4696569841451008",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/5111429943263232",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/6091125539471360",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4772943218343936",
        "https://ngl.cave-explorer.org/#!middleauth+https://global.daf-apis.com/nglstate/api/v1/4534147331653632",
    ],
}

for new_label, new_label_links in new_label_map.items():
    new_segments = []
    for link in new_label_links:
        state_id = int(link.split("/")[-1])
        state = client.state.get_state_json(state_id)
        segments = state["layers"][1]["segments"]
        segments = np.array([int(seg) for seg in segments if seg[0] != "!"])
        new_segments.extend(segments)

    new_segments = np.unique(new_segments)

    new_level2_ids = []
    for segment in tqdm(new_segments):
        leaves = client.chunkedgraph.get_leaves(segment, stop_layer=2)
        new_level2_ids.extend(leaves)

    columns = feature_df.columns.to_list()
    columns.remove("label")

    feature_path = Path("troglobyte-sandbox/results/vasculature")

    for filename in tqdm(glob.glob(str(feature_path / "vasculature_features_*.csv"))):
        sub_feature_df = pd.read_csv(filename, index_col=[0, 1])
        sub_feature_df = sub_feature_df[columns]

        sub_feature_df = sub_feature_df.loc[
            pd.IndexSlice[
                :,
                sub_feature_df.index.get_level_values("level2_id")
                .unique()
                .intersection(new_level2_ids),
            ],
            :,
        ]

        sub_feature_df["label"] = new_label

        new_features.append(sub_feature_df)


# %%
new_feature_df = pd.concat(new_features)
new_feature_df = new_feature_df.dropna()
new_feature_df = new_feature_df.loc[new_feature_df.index.drop_duplicates()]

combined_feature_df = pd.concat([feature_df, new_feature_df])
combined_feature_df = combined_feature_df.loc[
    combined_feature_df.index.drop_duplicates()
]
combined_feature_df = combined_feature_df.dropna()
combined_labels_by_level2 = combined_feature_df["label"]
combined_feature_df = combined_feature_df.drop(columns="label")

# %%
# nuc_table = client.materialize.query_table("nucleus_detection_v0")

cell_types_table = client.materialize.query_table("aibs_metamodel_celltypes_v661")

# %%
neuron_types = cell_types_table.query(
    "classification_system.isin(['excitatory_neuron', 'inhibitory_neuron'])"
)
nonneuron_types = cell_types_table.query("classification_system == 'nonneuron'")

# %%

from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors


def fit_neighbors_model(cell_types_table, n_neighbors=2):
    nuc_table = cell_types_table[["pt_position"]].copy()

    nuc_table["x"] = nuc_table["pt_position"].apply(lambda x: x[0])
    nuc_table["y"] = nuc_table["pt_position"].apply(lambda x: x[1])
    nuc_table["z"] = nuc_table["pt_position"].apply(lambda x: x[2])

    nuc_table["x"] = nuc_table["x"] * 4
    nuc_table["y"] = nuc_table["y"] * 4
    nuc_table["z"] = nuc_table["z"] * 40

    nuc_X = nuc_table[["x", "y", "z"]].values

    nuc_neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)

    nuc_neighbors.fit(nuc_X)

    return nuc_neighbors


def extract_positions(level2_ids, chunk_size=1000):
    level2_ids_by_chunk = np.array_split(level2_ids, len(level2_ids) // chunk_size)

    def get_xyz_for_chunk(chunk_level2_ids):
        out = client.l2cache.get_l2data(chunk_level2_ids, attributes=["rep_coord_nm"])
        out = pd.DataFrame(out).T
        out["x"] = out["rep_coord_nm"].apply(lambda x: x[0])
        out["y"] = out["rep_coord_nm"].apply(lambda x: x[1])
        out["z"] = out["rep_coord_nm"].apply(lambda x: x[2])
        out.index = out.index.astype(int)
        out = out[["x", "y", "z"]]
        return out

    outs_by_chunk: list = Parallel(n_jobs=-1, verbose=10)(
        delayed(get_xyz_for_chunk)(chunk_level2_ids)
        for chunk_level2_ids in level2_ids_by_chunk
    )

    level2_position_data = pd.concat(outs_by_chunk)

    return level2_position_data


def extract_neighbor_distances(neighbors_model, positions, label="nuc"):
    dists, nearest_nuc_idx = neighbors_model.kneighbors(
        positions.values, return_distance=True
    )
    n_neighbors = neighbors_model.n_neighbors
    dists = pd.DataFrame(
        dists,
        index=positions.index,
        columns=[f"{label}_dist_{i+1}" for i in range(n_neighbors)],
    )
    dists.index.name = "level2_id"
    return dists


neuron_nuc_neighbors = fit_neighbors_model(neuron_types)
nonneuron_nuc_neighbors = fit_neighbors_model(nonneuron_types)

level2_ids = combined_feature_df.index.get_level_values("level2_id").unique()

positions = extract_positions(level2_ids)

neuron_dists = extract_neighbor_distances(
    neuron_nuc_neighbors, positions, label="neuron_nuc"
)

nonneuron_dists = extract_neighbor_distances(
    nonneuron_nuc_neighbors, positions, label="nonneuron_nuc"
)

combined_feature_df = combined_feature_df.join(neuron_dists)
combined_feature_df = combined_feature_df.join(nonneuron_dists)


# %%
model = Pipeline(
    [
        ("transformer", QuantileTransformer(output_distribution="normal")),
        ("lda", LinearDiscriminantAnalysis()),
    ]
)

# %%
from sklearn.model_selection import train_test_split

train_inds, test_inds = train_test_split(
    combined_feature_df.index.get_level_values("object_id").unique(),
)
X_train = combined_feature_df.loc[train_inds]
y_train = combined_labels_by_level2.loc[train_inds]
X_test = combined_feature_df.loc[test_inds]
y_test = combined_labels_by_level2.loc[test_inds]

# %%
model.fit(X_train, y_train)
# %%
X_transformed = model.transform(combined_feature_df)

# %%
X_transformed_df = pd.DataFrame(data=X_transformed, index=combined_feature_df.index)
X_transformed_df.columns = [f"LDA{i}" for i in range(X_transformed_df.shape[1])]
X_transformed_df["label"] = combined_labels_by_level2

# %%

pg = sns.PairGrid(X_transformed_df, hue="label", corner=True)
pg.map_lower(sns.scatterplot, alpha=0.1, linewidth=0, s=1)
pg.map_diag(sns.histplot, kde=True)
pg.add_legend(markerscale=10)

# %%

print("Train:")
y_pred_train = model.predict(X_train)
print(classification_report(y_train, y_pred_train))
print()
print("Test:")
y_pred_test = model.predict(X_test)
print(classification_report(y_test, y_pred_test))
print()
print("All:")
y_pred_all = model.predict(combined_feature_df)
print(classification_report(combined_labels_by_level2, y_pred_all))

# %%
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred_test, labels=model.classes_)

conf_mat = pd.DataFrame(conf_mat, index=model.classes_, columns=model.classes_)
conf_mat.index.name = "true"
conf_mat.columns.name = "predicted"
conf_mat

# %%
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=500, max_depth=5, n_jobs=-1)

model_rf.fit(X_train, y_train)

print("Train:")
y_pred_train = model_rf.predict(X_train)
print(classification_report(y_train, y_pred_train))
print()
print("Test:")
y_pred_test = model_rf.predict(X_test)
print(classification_report(y_test, y_pred_test))
print()

conf_mat = confusion_matrix(y_test, y_pred_test, labels=model.classes_)

conf_mat = pd.DataFrame(conf_mat, index=model_rf.classes_, columns=model_rf.classes_)
conf_mat.index.name = "true"
conf_mat.columns.name = "predicted"
conf_mat


# %%
for filename in tqdm(glob.glob(str(feature_path / "vasculature_features_*.csv"))[0:]):
    print(filename)
    vasculature_feature_df = pd.read_csv(filename, index_col=[0, 1])
    vasculature_feature_df = vasculature_feature_df[columns]
    break

box = filename.split("=")[1].split("_")
box = [int(x.strip(".csv")) for x in box]
box


# %%
box_name = "_".join([str(x) for x in box])

# %%

import pickle

# for filename in tqdm(glob.glob(str(feature_path / "wrangler_*.pkl"))[:]):
filename = feature_path / f"wrangler_box={box_name}.pkl"
with open(filename, "rb") as f:
    wrangler = pickle.load(f)
    wrangler.client = client


# %%
lumen_segments = [
    377858200361292074,
    377862598407805983,
    378421150314723072,
    378984100268135823,
    522910137985918230,
]

cv = client.info.segmentation_cloudvolume()

lumen_meshes = cv.mesh.get(lumen_segments)

# %%

import pyvista as pv

from neurovista import to_mesh_polydata

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

padded_bbox_pyvista = bounds_to_pyvista(wrangler.query_box_)
padded_bbox_mesh = pv.Box(bounds=padded_bbox_pyvista)
plotter.add_mesh(padded_bbox_mesh, color="blue", style="wireframe")

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
# points = np.concatenate(points)

# %%
points.shape

neighbors = NearestNeighbors(n_neighbors=1, n_jobs=-1)
neighbors.fit(points)

# %%
new_positions = extract_positions(
    vasculature_feature_df.index.get_level_values("level2_id").unique()
)

neuron_dists = extract_neighbor_distances(
    neuron_nuc_neighbors, new_positions, label="neuron_nuc"
)
nonneuron_dists = extract_neighbor_distances(
    nonneuron_nuc_neighbors, new_positions, label="nonneuron_nuc"
)

vasculature_feature_df = vasculature_feature_df.join(neuron_dists)
vasculature_feature_df = vasculature_feature_df.join(nonneuron_dists)
vasculature_feature_df = vasculature_feature_df.join(new_positions)


# %%

xmin = bbox[0, 0]
ymin = bbox[0, 1]
zmin = bbox[0, 2]
xmax = bbox[1, 0]
ymax = bbox[1, 1]
zmax = bbox[1, 2]

truncate = True
if truncate:
    vasculature_feature_df = vasculature_feature_df.query(
        "x >= @xmin & x <= @xmax & y >= @ymin & y <= @ymax & z >= @zmin & z <= @zmax"
    )


# %%
vasculature_X = vasculature_feature_df.dropna()[model.feature_names_in_]

# %%
vasculature_pred = model_rf.predict(vasculature_X)

vasculature_X_transformed = model.transform(vasculature_X)

# %%
vasculature_X_transformed_df = pd.DataFrame(
    data=vasculature_X_transformed, index=vasculature_X.index
)
vasculature_X_transformed_df.columns = [
    f"LDA{i}" for i in range(vasculature_X_transformed_df.shape[1])
]
vasculature_X_transformed_df["label"] = vasculature_pred

pg = sns.PairGrid(vasculature_X_transformed_df, hue="label", corner=True)
pg.map_lower(sns.scatterplot, alpha=0.2, linewidth=0, s=1)
pg.map_diag(sns.histplot, kde=True)
pg.add_legend(markerscale=20)

# %%

dists, neighbor_indices = neighbors.kneighbors(
    vasculature_feature_df[["x", "y", "z"]].values, return_distance=True
)

vasculature_feature_df["lumen_min_dist"] = dists

# %%
Z = (
    vasculature_X.join(
        pd.Series(vasculature_pred, index=vasculature_X.index, name="label")
    )
    .query("label == 'perivascular'")
    .drop(columns="label")
)

Z_trans = model.steps[0][1].transform(Z)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z_trans)

for i in range(1):
    plt.figure()
    plt.scatter(Z_pca[:, i], Z_pca[:, i + 1])
    plt.xlabel(f"PCA{i}")
    plt.ylabel(f"PCA{i+1}")
    plt.show()

# %%
Z[Z_pca[:, 0] > 0].index.get_level_values("level2_id").unique().to_series().to_list()

# %%
#
test_ids = [
    161933942906159894,
    162144980419216158,
    162215005566010214,
    162215005633118223,
    162215143004963270,
    162285374310188083,
    162285374377295890,
    162285443029664462,
    162285443096772609,
    162285511749140826,
    162285511749141221,
    162285511749141225,
    162285511749141267,
    162285580468617611,
    162285717907571494,
    162355674334888585,
    162355743054365400,
    162355743054365405,
    162355811773841897,
    162355811773842144,
    162355811773842154,
    162355811773842158,
    162355880493318840,
    162355880493318847,
    162355880493318853,
    162355949212795670,
    162356017932272331,
    162356086651748906,
    162356086718857223,
    162425974359589458,
    162426043079066094,
    162426043079066312,
    162426111798543017,
    162426111798543024,
    162426180518019743,
    162426180518019765,
    162426249237496467,
    162426317956973260,
    162496343103767209,
    162496411823243953,
    162496480542720574,
    162496617981674147,
    162496686701150858,
    162566986725851908,
    162567055445328551,
    162637424189506119,
    162707792933683836,
]

inds = Z.index.droplevel("object_id").get_indexer_for(test_ids)

colors = np.array(len(Z_trans) * ["black"])
colors[inds] = "red"

pca = PCA(n_components=10)
Z_pca = pca.fit_transform(Z_trans)

for i in range(9):
    plt.figure()
    sns.scatterplot(x=Z_pca[:, i], y=Z_pca[:, i + 1], hue=colors)
    plt.xlabel(f"PCA{i}")
    plt.ylabel(f"PCA{i+1}")
    plt.show()


# %%
vasculature_X.index[Z_pca[:, 0] > 0]


# %%


vasculature_pred_df = pd.DataFrame(
    data=vasculature_pred, index=vasculature_X.index, columns=["label"]
)
vasculature_pred_df["lumen_min_dist"] = vasculature_feature_df["lumen_min_dist"]

# level2_counts_by_root = vasculature_pred_df.groupby("object_id").size()
# big_roots = level2_counts_by_root[level2_counts_by_root > 5].index
# big_vasculature_pred_df = vasculature_pred_df.loc[big_roots]

sub_vasculature_index = (
    vasculature_pred_df.index.get_level_values("object_id").unique().to_series()
    # .sample(100)
)
sub_vasculature_pred_df = vasculature_pred_df.loc[sub_vasculature_index].copy()


level2_ids = sub_vasculature_pred_df.index.get_level_values("level2_id")

# %%

timestamp = client.materialize.get_timestamp(client.materialize.version)

# %%

max_level = 6
min_level = 3
for level in range(min_level, max_level + 1):
    level_ids = client.chunkedgraph.get_roots(
        level2_ids, stop_layer=level, timestamp=timestamp
    )
    sub_vasculature_pred_df[f"level{level}_id"] = level_ids

# %%
sub_vasculature_pred_df = sub_vasculature_pred_df.reset_index()

# %%

level_names = [f"level{level}_id" for level in range(min_level, max_level + 1)]
level_names += ["object_id"]

mixed_level_df = []
for level_name in level_names:
    groupby = sub_vasculature_pred_df.groupby(level_name)
    label_counts = groupby["label"].value_counts()
    label_props = label_counts / label_counts.groupby(level_name).transform("sum")
    label_counts = label_counts.unstack().fillna(0)
    label_props = label_props.unstack().fillna(0)
    label_props = label_props.rename(columns=lambda x: f"{x}_prop")
    dists = groupby["lumen_min_dist"].min()

    new_df = pd.concat([label_props, dists], axis=1)
    new_df.index.name = "node_id"
    new_df.index = new_df.index.astype("Int64")
    if level_name == "object_id":
        level_name = "root_id"
    new_df["level"] = level_name
    mixed_level_df.append(new_df.reset_index())

mixed_level_df = pd.concat(mixed_level_df)
# mixed_level_df.index = mixed_level_df.index.astype("int64")


# %%


import numpy as np
from nglui.segmentprops import SegmentProperties

seg_df = mixed_level_df.copy()
# seg_df = seg_df.sample(20000)

n_randoms = 3
for i in range(n_randoms):
    seg_df[f"random_{i}"] = np.random.uniform(0, 1, size=len(seg_df))


seg_prop = SegmentProperties.from_dataframe(
    seg_df.reset_index(),
    id_col="node_id",
    label_col="node_id",
    tag_value_cols=["level"],
    number_cols=[f"random_{i}" for i in range(n_randoms)]
    + ["lumen_min_dist"]
    + [f"{cl}_prop" for cl in model.classes_],
)

prop_id = client.state.upload_property_json(seg_prop.to_dict())
prop_url = client.state.build_neuroglancer_url(
    prop_id, format_properties=True, target_site="mainline"
)

from nglui import statebuilder

client = CAVEclient("minnie65_public")
client.materialize.version = 1078

img = statebuilder.ImageLayerConfig(
    source=client.info.image_source(),
)
seg = statebuilder.SegmentationLayerConfig(
    source=client.info.segmentation_source(),
    segment_properties=prop_url,
    fixed_ids=lumen_segments,
    active=True,
    skeleton_source="precomputed://middleauth+https://minnie.microns-daf.com/skeletoncache/api/v1/minnie65_phase3_v1/precomputed/skeleton",
)

sb = statebuilder.StateBuilder(
    layers=[img, seg],
    target_site="mainline",
    view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001},
)

sb.render_state()

# %%

level = 6
query_df = mixed_level_df.query(f"level == 'level{level}_id' & axon_prop > 0.8")

node_ids = query_df["node_id"].values
# %%
import time

currtime = time.time()
sample_meshes = cv.mesh.get(node_ids[:100], deduplicate_chunk_boundaries=False)
print(f"{time.time() - currtime:.3f} seconds elapsed.")


# %%
def download_mesh_chunk(chunk_nodes):
    return cv.mesh.get(
        chunk_nodes, deduplicate_chunk_boundaries=False, allow_missing=True
    )


n_per_chunk = 50
n_chunks = np.ceil(len(node_ids) / n_per_chunk).astype(int)
chunked_node_ids = np.array_split(node_ids, n_chunks)


currtime = time.time()
axon_meshes = Parallel(n_jobs=-1, verbose=10)(
    delayed(download_mesh_chunk)(chunk_nodes) for chunk_nodes in chunked_node_ids
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
all_meshes = {}
for mesh_group in axon_meshes:
    all_meshes.update(mesh_group)

# %%

pv.set_jupyter_backend("client")

bbox_mesh = pv.Box(bounds=bbox_pyvista)

polys = []
for i, mesh in tqdm(enumerate(all_meshes.values()), total=len(all_meshes)):
    if np.random.rand() < 0.2:
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
plotter = pv.Plotter()

plotter.open_gif("axon_meshes.gif", fps=30)

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

# plotter.show()

# %%

# # %%
# current_level = max_level
# current_df = sub_vasculature_pred_df

# labels_by_level = []

# for current_level in range(max_level, 1, -1):
#     label_counts_at_level = current_df.groupby([f"level{current_level}_id"])[
#         "label"
#     ].nunique()
#     singletons = label_counts_at_level[label_counts_at_level == 1].index
#     new_labels = (
#         current_df.set_index(f"level{current_level}_id")
#         .loc[singletons]
#         .groupby([f"level{current_level}_id"])["label"]
#         .first()
#     )
#     new_labels = new_labels.to_frame()
#     new_labels["level"] = current_level
#     new_labels.index = new_labels.index.astype("uint64")
#     labels_by_level.append(new_labels)
#     current_df = current_df.set_index(f"level{current_level}_id").drop(singletons)
#     break
# # %%
# labels_by_level = pd.concat(labels_by_level)
# labels_by_level.index.name = "node_id"
# # %%

# import numpy as np
# from nglui.segmentprops import SegmentProperties

# seg_df = labels_by_level.copy()
# # seg_df = seg_df.sample(20000)

# n_randoms = 3
# for i in range(n_randoms):
#     seg_df[f"random_{i}"] = np.random.uniform(0, 1, size=len(seg_df))

# seg_prop = SegmentProperties.from_dataframe(
#     seg_df.reset_index(),
#     id_col="node_id",
#     label_col="label",
#     tag_value_cols="label",
#     number_cols=[f"random_{i}" for i in range(n_randoms)],
# )

# prop_id = client.state.upload_property_json(seg_prop.to_dict())
# prop_url = client.state.build_neuroglancer_url(
#     prop_id, format_properties=True, target_site="mainline"
# )

# from nglui import statebuilder

# client = CAVEclient("minnie65_public")
# client.materialize.version = 1078

# img = statebuilder.ImageLayerConfig(
#     source=client.info.image_source(),
# )
# seg = statebuilder.SegmentationLayerConfig(
#     source=client.info.segmentation_source(),
#     segment_properties=prop_url,
#     fixed_ids=seg_df.index[0],
#     active=True,
# )

# sb = statebuilder.StateBuilder(
#     layers=[img, seg],
#     target_site="mainline",
#     view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001},
# )

# sb.render_state()

# # %%

# model_0 = Pipeline(
#     [
#         ("transformer", QuantileTransformer(output_distribution="normal")),
#         ("lda", LinearDiscriminantAnalysis()),
#     ]
# )

# hierarchical_labels = combined_labels_by_level2.to_frame().copy()

# hierarchical_labels["0"] = hierarchical_labels["label"].apply(
#     lambda x: True if x in ["axon", "dendrite"] else False
# )
# hierarchical_labels["1"] = hierarchical_labels["label"].copy()

# y_train = hierarchical_labels.loc[train_inds]
# y_test = hierarchical_labels.loc[test_inds]

# model_0.fit(X_train, y_train["0"])
# y_pred_train = model_0.predict(X_train)

# print("Train:")
# print(classification_report(y_train["0"], y_pred_train))


# # %%

# sub_labels = combined_labels_by_level2[
#     combined_labels_by_level2.isin(["soma", "perivascular"])
# ]
# sub_labels = sub_labels[sub_labels.index.drop_duplicates()]
# sub_feature_df = combined_feature_df.loc[sub_labels.index]
# sub_feature_df = sub_feature_df.loc[sub_feature_df.index.drop_duplicates()]


# # %%
# model_big = Pipeline(
#     [
#         ("transformer", QuantileTransformer(output_distribution="normal")),
#         ("lda", LinearDiscriminantAnalysis()),
#     ]
# )

# model_big.fit(sub_feature_df, sub_labels)

# sub_labels_pred = model_big.predict(sub_feature_df)
