# %%


from pathlib import Path

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from skops.io import dump

client = cc.CAVEclient("minnie65_phase3_v1")

out_path = Path("./troglobyte-sandbox/models/")

model_name = "local_compartment_classifier_bd_boxes"

data_path = Path("./troglobyte-sandbox/data/bounding_box_labels")

files = list(data_path.glob("*.csv"))

# %%
label_df = pd.read_csv(out_path / model_name / "labels.csv", index_col=[0, 1])
label_df = label_df.rename(columns=lambda x: x.replace(".1", ""))

# # %%

# X_df = wrangler.features_.copy()
# X_df = X_df.drop(columns=[col for col in X_df.columns if "rep_coord" in col])

# %%

X_df = pd.read_csv(out_path / model_name / "features_new.csv", index_col=[0, 1])


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


# models to evaluate
def get_lda(n_classes):
    lda = Pipeline(
        [
            ("transformer", QuantileTransformer(output_distribution="normal")),
            ("lda", LinearDiscriminantAnalysis(n_components=n_classes - 1)),
        ]
    )
    return lda


rf = RandomForestClassifier(n_estimators=500, max_depth=4)

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
        for model_type, model in models.items():
            model.fit(train_X_df, train_l2_y)
            train_preds = model.predict(train_X_df)
            test_preds = model.predict(test_X_df)

            # evaluate at the L2 level
            train_report = classification_report(
                train_l2_y, train_preds, output_dict=True
            )
            rows.append(
                {
                    "model": model_type,
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
                    "model": model_type,
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
                    "model": model_type + "-vote",
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
                    "model": model_type + "-vote",
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
final_lda = Pipeline(
    [
        ("transformer", QuantileTransformer(output_distribution="normal")),
        ("lda", LinearDiscriminantAnalysis(n_components=n_classes - 1)),
    ]
)

train_X_df, test_X_df, train_l2_y, test_l2_y = box_train_test_split(
    np.array([0, 1, 2]), np.array([]), X_df, label_df, label_column
)

final_lda.fit(train_X_df, train_l2_y)

report = classification_report(
    train_l2_y, final_lda.predict(train_X_df), output_dict=True
)

# %%
report_table = pd.DataFrame(report).T

report_overall = report_table.loc[["accuracy", "macro avg", "weighted avg"]]
report_overall.index.name = "type"
report_overall.reset_index(inplace=True)
report_by_class = report_table.drop(index=["accuracy", "macro avg", "weighted avg"])
report_by_class.index.name = "class"
report_by_class.reset_index(inplace=True)
# %%

# model_pickle_file = out_path / model_name / f"{model_name}.skops"
# with open(model_pickle_file, mode="bw") as f:
#     dump(final_lda, file=f)

# %%
import os
from pathlib import Path

from skops import card, hub_utils

hub_out_path = Path(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/hub_model"
)
if not hub_out_path.exists():
    hub_utils.init(
        model=model_pickle_file,
        requirements=["scikit-learn", "caveclient"],
        dst=hub_out_path,
        task="tabular-classification",
        data=train_X_df,
    )

hub_utils.add_files(__file__, dst=hub_out_path, exist_ok=True)

# if True:
if not os.path.exists(hub_out_path / "README.md"):
    model_card = card.Card(model, metadata=card.metadata_from_config(hub_out_path))
    model_card.metadata.license = "mit"
    model_description = (
        "This is a model trained to classify pieces of neuron as axon, dendrite, soma, or"
        "glia, "
        "based only on their local shape and synapse features."
        "The model is a linear discriminant classifier which was trained on compartment "
        "labels generated by Bethanny Danskin for 3 6x6x6 um boxes in the Minnie65 Phase3 "
        "dataset."
    )
    model_card_authors = "bdpedigo"
    model_card.add(
        model_card_authors=model_card_authors,
        model_description=model_description,
    )
    model_card.add_table(
        folded=False,
        **{
            "Classification Report (overall)": report_overall,
            "Classification Report (by class)": report_by_class,
        },
    )
    model_card.save(hub_out_path / "README.md")

from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    repo_id=f"bdpedigo/{model_name}",
    folder_path=hub_out_path,
    # filename=f"{model_name}.skops",
)
# hub_utils.push(
#     repo_id=f"bdpedigo/{model_name}",
#     source=hub_out_path,
#     create_remote=False,
#     private=False,
# )

# %%

syn_features = [col for col in X_df.columns if "syn" in col]
train_X_df_no_syn = train_X_df.drop(columns=syn_features)

final_lda_no_syn = Pipeline(
    [
        ("transformer", QuantileTransformer(output_distribution="normal")),
        ("lda", LinearDiscriminantAnalysis(n_components=n_classes - 1)),
    ]
)

final_lda_no_syn.fit(train_X_df_no_syn, train_l2_y)

print(classification_report(train_l2_y, final_lda_no_syn.predict(train_X_df_no_syn)))

with open(out_path / model_name / f"{model_name}_no_syn.skops", mode="bw") as f:
    dump(final_lda_no_syn, file=f)
