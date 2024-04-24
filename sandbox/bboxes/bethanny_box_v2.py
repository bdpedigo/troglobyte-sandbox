# %%


from pathlib import Path

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from troglobyte.features import L2AggregateWrangler

client = cc.CAVEclient("minnie65_phase3_v1")

# %%


data_path = Path("./troglobyte-sandbox/data/bounding_box_labels")

files = list(data_path.glob("*.csv"))

# %%
voxel_resolution = np.array([4, 4, 40])


def simple_labeler(label):
    if "axon" in label:
        return "axon"
    elif "dendrite" in label:
        return "dendrite"
    elif "glia" in label:
        return "glia"
    elif "soma" in label:
        return "soma"
    else:
        return np.nan


def axon_labeler(label):
    if label == "uncertain":
        return np.nan
    elif "axon" in label:
        return True
    else:
        return False


dfs = []
for file in files:
    file_label_df = pd.read_csv(file)

    file_label_df.set_index(["bbox_id", "root_id"], inplace=True, drop=False)
    file_label_df["ctr_pt_4x4x40"] = file_label_df["ctr_pt_4x4x40"].apply(
        lambda x: np.array(eval(x.replace("  ", ",").replace(" ", ",")), dtype=int)
    )

    file_label_df["x_nm"] = (
        file_label_df["ctr_pt_4x4x40"].apply(lambda x: x[0]) * voxel_resolution[0]
    )
    file_label_df["y_nm"] = (
        file_label_df["ctr_pt_4x4x40"].apply(lambda x: x[1]) * voxel_resolution[1]
    )
    file_label_df["z_nm"] = (
        file_label_df["ctr_pt_4x4x40"].apply(lambda x: x[2]) * voxel_resolution[2]
    )

    file_label_df["axon_label"] = file_label_df["label"].apply(axon_labeler)

    file_label_df["simple_label"] = file_label_df["label"].apply(simple_labeler)

    dfs.append(file_label_df)

# %%
label_df = pd.concat(dfs)


# %%

points = label_df[["x_nm", "y_nm", "z_nm"]].values
neighborhood_hops = 10
feature_extractor = L2AggregateWrangler(
    client,
    verbose=10,
    n_jobs=-1,
    neighborhood_hops=neighborhood_hops,
    box_width=10_000,
)
node_features = feature_extractor.get_features(label_df["root_id"], domains=points)


# %%

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

X_df = node_features.copy()

drop_syn_features = False
drop_pca_vec_features = False
drop_pca_val_features = False
drop_rep_coord_features = True
drop_scalar_features = False
if drop_syn_features:
    X_df = X_df.drop(columns=syn_features)
if drop_pca_vec_features:
    X_df = X_df.drop(columns=pca_vec_features)
if drop_pca_val_features:
    X_df = X_df.drop(columns=pca_val_features)
if drop_rep_coord_features:
    X_df = X_df.drop(columns=rep_coord_features)
if drop_scalar_features:
    X_df = X_df.drop(columns=scalar_features)


# %%


def box_train_test_split(
    train_box_indices, test_box_indices, X_df, label_df, label_column
):
    train_label_df = label_df.loc[train_box_indices + 1].droplevel("bbox_id")
    test_label_df = label_df.loc[test_box_indices + 1].droplevel("bbox_id")

    train_X_df = X_df.loc[train_label_df["root_id"]]
    test_X_df = X_df.loc[test_label_df["root_id"]]
    train_X_df = train_X_df.dropna()
    test_X_df = test_X_df.dropna()

    train_l2_y = train_X_df.index.get_level_values("object_id").map(
        train_label_df[label_column]
    )
    test_l2_y = test_X_df.index.get_level_values("object_id").map(
        test_label_df[label_column]
    )

    # TODO do something more fair here w/ evaluation on the uncertains
    train_X_df = train_X_df.loc[train_l2_y.notna()]
    train_l2_y = train_l2_y[train_l2_y.notna()].values.astype(str)

    test_X_df = test_X_df.loc[test_l2_y.notna()]
    test_l2_y = test_l2_y[test_l2_y.notna()].values.astype(str)

    return train_X_df, test_X_df, train_l2_y, test_l2_y


# %%


def aggregate_votes_by_object(X_df, l2_node_predictions):
    l2_node_predictions = pd.Series(
        index=X_df.index, data=l2_node_predictions, name="label"
    )

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
    max_predictions = max_predictions.set_index("object_id")
    return max_predictions, object_prediction_probs


rf = RandomForestClassifier(n_estimators=100, max_depth=3)


def get_lda(n_classes):
    lda = Pipeline(
        [
            ("transformer", QuantileTransformer(output_distribution="normal")),
            ("lda", LinearDiscriminantAnalysis(n_components=n_classes - 1)),
        ]
    )
    return lda


box_indices = np.arange(1, 4)

rows = []


for fold, (train_box_indices, test_box_indices) in enumerate(
    KFold(n_splits=3).split(box_indices.reshape(-1, 1))
):
    for label_column in ["axon_label", "simple_label"]:
        train_X_df, test_X_df, train_l2_y, test_l2_y = box_train_test_split(
            train_box_indices, test_box_indices, X_df, label_df, label_column
        )
        n_classes = label_df[label_column].nunique()
        models = {"rf": rf, "lda": get_lda(n_classes)}
        for model_name, model in models.items():
            model.fit(train_X_df, train_l2_y)
            train_preds = model.predict(train_X_df)
            test_preds = model.predict(test_X_df)

            # evaluate at the L2 level
            train_report = classification_report(
                train_l2_y, train_preds, output_dict=True
            )
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "accuracy": train_report["accuracy"],
                    "macro_f1": train_report["macro avg"]["f1-score"],
                    "weighted_f1": train_report["weighted avg"]["f1-score"],
                    "evaluation": "train",
                    "labeling": label_column,
                    "level": "level2",
                }
            )

            test_report = classification_report(test_l2_y, test_preds, output_dict=True)
            rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "accuracy": test_report["accuracy"],
                    "macro_f1": test_report["macro avg"]["f1-score"],
                    "weighted_f1": test_report["weighted avg"]["f1-score"],
                    "evaluation": "test",
                    "labeling": label_column,
                    "level": "level2",
                }
            )

            # evaluate at the object level
            train_object_predictions, train_object_probs = aggregate_votes_by_object(
                train_X_df, train_preds
            )
            train_object_y = (
                label_df.droplevel(0)
                .loc[train_object_predictions.index, label_column]
                .values.astype(str)
            )
            train_object_report = classification_report(
                train_object_y, train_object_predictions["label"], output_dict=True
            )
            rows.append(
                {
                    "model": model_name + "-vote",
                    "fold": fold,
                    "accuracy": train_object_report["accuracy"],
                    "macro_f1": train_object_report["macro avg"]["f1-score"],
                    "weighted_f1": train_object_report["weighted avg"]["f1-score"],
                    "evaluation": "train",
                    "labeling": label_column,
                    "level": "root",
                }
            )

            test_object_predictions, test_object_probs = aggregate_votes_by_object(
                test_X_df, test_preds
            )
            test_object_y = (
                label_df.droplevel(0)
                .loc[test_object_predictions.index, label_column]
                .values.astype(str)
            )
            test_object_report = classification_report(
                test_object_y, test_object_predictions["label"], output_dict=True
            )
            rows.append(
                {
                    "model": model_name + "-vote",
                    "fold": fold,
                    "accuracy": test_object_report["accuracy"],
                    "macro_f1": train_object_report["macro avg"]["f1-score"],
                    "weighted_f1": train_object_report["weighted avg"]["f1-score"],
                    "evaluation": "test",
                    "labeling": label_column,
                    "level": "root",
                }
            )


# %%


evaluation_df = pd.DataFrame(rows)

sns.set_context("talk")

fig, axs = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True, sharey="col")

for i, labeling in enumerate(["simple_label", "axon_label"]):
    for j, metric in enumerate(["accuracy", "weighted_f1", "macro_f1"]):
        ax = axs[i, j]
        show_legend = (i == 0) & (j == 0)
        sns.stripplot(
            data=evaluation_df.query("labeling == @labeling"),
            x="model",
            y=metric,
            hue="evaluation",
            ax=ax,
            legend=show_legend,
            s=10,
            jitter=True,
        )
        ax.spines[["right", "top"]].set_visible(False)
        if j == 1:
            ax.set_title("Labeling: " + labeling)


# %%
lda = model
train_X_transformed = lda.transform(train_X_df)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    x=train_X_transformed[:, 0],
    y=train_X_transformed[:, 1],
    hue=train_l2_y,
    ax=ax,
    s=10,
    alpha=0.7,
)
ax.set(xticks=[], yticks=[], xlabel="LDA1", ylabel="LDA2")
ax.spines[["right", "top"]].set_visible(False)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    x=train_X_transformed[:, 0],
    y=train_X_transformed[:, 2],
    hue=train_l2_y,
    ax=ax,
    s=10,
    alpha=0.7,
)
ax.set(xticks=[], yticks=[], xlabel="LDA1", ylabel="LDA3")
ax.spines[["right", "top"]].set_visible(False)

# %%
pca = PCA(n_components=3)

train_X_pca = pca.fit_transform(train_X_df)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.scatterplot(
    x=train_X_pca[:, 0],
    y=train_X_pca[:, 2],
    hue=train_l2_y,
    ax=ax,
    s=10,
    alpha=0.7,
)
ax.set(xticks=[], yticks=[], xlabel="PCA1", ylabel="PCA2")
ax.spines[["right", "top"]].set_visible(False)
# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 12))
sns.heatmap(lda.steps[1][1].coef_.T, ax=ax, cmap="RdBu_r", center=0)
ax.set_yticks(np.arange(len(train_X_df.columns)) + 0.5)
ax.set_yticklabels(train_X_df.columns)

# %%
final_lda = Pipeline(
    [
        ("transformer", QuantileTransformer(output_distribution="normal")),
        ("lda", LinearDiscriminantAnalysis(n_components=n_classes - 1)),
    ]
)

# %%

train_X_df, test_X_df, train_l2_y, test_l2_y = box_train_test_split(
    np.array([0, 1, 2]), np.array([]), X_df, label_df, label_column
)

# %%

final_lda.fit(train_X_df, train_l2_y)

# %%
from joblib import dump

dump(final_lda, "bethanny_box_model=quantile_lda_train=1-2-3_.joblib")

# %%

train_X_df_no_syn = train_X_df.drop(columns=syn_features)

final_lda_no_syn = Pipeline(
    [
        ("transformer", QuantileTransformer(output_distribution="normal")),
        ("lda", LinearDiscriminantAnalysis(n_components=n_classes - 1)),
    ]
)

final_lda_no_syn.fit(train_X_df_no_syn, train_l2_y)

dump(final_lda_no_syn, "bethanny_box_model=quantile_lda_train=1-2-3_no_syn.joblib")
