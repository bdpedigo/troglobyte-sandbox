# %%

import pprint

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.preprocessing import QuantileTransformer

from troglobyte.features import L2AggregateWrangler

client = cc.CAVEclient("minnie65_phase3_v1")


# %%
voxel_resolution = np.array([4, 4, 40])
label_df = pd.read_csv(
    "troglobyte-examples/data/bd_subvolume_segment_labels_240409.csv"
)

label_df = (
    label_df[["Segment IDs", "Tags", "Coordinate 1"]]
    .rename(
        columns={
            "Segment IDs": "segment_id",
            "Tags": "tags",
            "Coordinate 1": "point_voxels",
        }
    )
    .set_index("segment_id")
)
label_df["point_voxels"] = (
    label_df["point_voxels"].str.replace("(", "").str.replace(")", "")
)
label_df["x_voxels"] = label_df["point_voxels"].apply(lambda x: x.split(",")[0])
label_df["y_voxels"] = label_df["point_voxels"].apply(lambda x: x.split(",")[1])
label_df["z_voxels"] = label_df["point_voxels"].apply(lambda x: x.split(",")[2])
label_df["x_nm"] = label_df["x_voxels"].astype(int) * voxel_resolution[0]
label_df["y_nm"] = label_df["y_voxels"].astype(int) * voxel_resolution[1]
label_df["z_nm"] = label_df["z_voxels"].astype(int) * voxel_resolution[2]

label_df["axon_label"] = label_df["tags"].str.contains("axon")


# %%

points = label_df[["x_nm", "y_nm", "z_nm"]].values
neighborhood_hops = 10
feature_extractor = L2AggregateWrangler(
    client,
    verbose=10,
    n_jobs=8,
    neighborhood_hops=neighborhood_hops,
    box_width=10_000,
)
node_features = feature_extractor.get_features(label_df.index, domains=points)


# %%
node_features

syn_features = ["pre_synapse_count", "post_synapse_count"]
pca_vec_features = [f"pca_unwrapped_{i}" for i in range(9)]
pca_val_features = [f"pca_val_unwrapped_{i}" for i in range(3)]
pca_val_features += ["pca_val_ratio"]
rep_coord_features = ["rep_coord_x", "rep_coord_y", "rep_coord_z"]
scalar_features = ["area_nm2", "max_dt_nm", "mean_dt_nm", "size_nm3"]

self_features = (
    syn_features
    + pca_vec_features
    + pca_val_features
    + rep_coord_features
    + scalar_features
).copy()

syn_features = syn_features + [f"{feature}_neighbor_agg" for feature in syn_features]
pca_vec_features = pca_vec_features + [
    f"{feature}_neighbor_agg" for feature in pca_vec_features
]
pca_val_features = pca_val_features + [
    f"{feature}_neighbor_agg" for feature in pca_val_features
]
rep_coord_features = rep_coord_features + [
    f"{feature}_neighbor_agg" for feature in rep_coord_features
]
scalar_features = scalar_features + [
    f"{feature}_neighbor_agg" for feature in scalar_features
]
neighbor_features = np.setdiff1d(node_features.columns, self_features)

# %%

drop_syn_features = False
drop_pca_vec_features = False
drop_pca_val_features = False
drop_rep_coord_features = True
drop_scalar_features = False

select_X_df = node_features
if drop_syn_features:
    select_X_df = select_X_df.drop(columns=syn_features)
if drop_pca_vec_features:
    select_X_df = select_X_df.drop(columns=pca_vec_features)
if drop_pca_val_features:
    select_X_df = select_X_df.drop(columns=pca_val_features)
if drop_rep_coord_features:
    select_X_df = select_X_df.drop(columns=rep_coord_features)
if drop_scalar_features:
    select_X_df = select_X_df.drop(columns=scalar_features)


# %%

X_preprocessed = select_X_df
transform = True
if transform:
    transformer = QuantileTransformer(output_distribution="normal")
    X_preprocessed = transformer.fit_transform(select_X_df)
    X_preprocessed = pd.DataFrame(
        X_preprocessed,
        columns=select_X_df.columns,
        index=select_X_df.index,
    )
else:
    X_preprocessed = X_preprocessed.copy()

X_preprocessed = X_preprocessed.dropna()

# %%
label_column = "axon_label"
n_labels = label_df[label_column].nunique()

lda = LinearDiscriminantAnalysis(n_components=1)

node_labels = X_preprocessed.index.get_level_values("object_id").map(
    label_df[label_column]
)

# %%
node_features_transformed = lda.fit_transform(X_preprocessed, node_labels)


# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    x=node_features_transformed[:, 0],
    y=node_features_transformed[:, 1],
    hue=node_labels,
    ax=ax,
    s=10,
    alpha=0.7,
)
ax.set(xticks=[], yticks=[])

# %%
l2_node_predictions = lda.predict(X_preprocessed)

# %%
l2_node_predictions = pd.Series(
    index=X_preprocessed.index, data=l2_node_predictions, name="compartment"
)
# %%
object_prediction_counts = (
    l2_node_predictions.groupby(level="object_id").value_counts().to_frame()
)

object_n_predictions = object_prediction_counts.groupby("object_id").sum()

sufficient_data_index = object_n_predictions.query("count > 3").index

object_prediction_counts = object_prediction_counts.loc[sufficient_data_index]

object_prediction_probs = object_prediction_counts.unstack(fill_value=0)
object_prediction_probs = object_prediction_probs.div(
    object_prediction_probs.sum(axis=1), axis=0
)


object_prediction_counts.reset_index(drop=False, inplace=True)


max_locs = object_prediction_counts.groupby("object_id")["count"].idxmax()

max_predictions = object_prediction_counts.loc[max_locs]
max_predictions["proportion"] = (
    max_predictions["count"]
    / object_n_predictions.loc[max_predictions["object_id"]]["count"].values
)

# %%

report = classification_report(
    label_df[label_column].loc[max_predictions["object_id"]],
    max_predictions["compartment"],
    output_dict=True,
)


pprint.pprint(report)

# %%
feature_extractor = L2AggregateWrangler(
    client,
    verbose=10,
    n_jobs=8,
    neighborhood_hops=neighborhood_hops,
    box_width=10_000,
)
networkframes = feature_extractor.get_networks(label_df.index[:5], domains=points[:5])
# %%
nf = networkframes.iloc[0]
nodes = nf.nodes
pos = nodes[["rep_coord_x", "rep_coord_y", "rep_coord_z"]]
point_pos = label_df.iloc[0][["x_nm", "y_nm", "z_nm"]].rename(
    {"x_nm": "rep_coord_x", "y_nm": "rep_coord_y", "z_nm": "rep_coord_z"}
)
# %%
