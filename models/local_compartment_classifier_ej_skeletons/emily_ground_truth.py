# %%
import io
import re
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skops
import skops.hub_utils
from caveclient import CAVEclient
from cloudfiles import CloudFiles
from meshparty import meshwork
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from skops import card
from skops.io import dump
from tqdm.auto import tqdm

from troglobyte.features import CAVEWrangler


def load_mw(directory, filename):
    # REF: stolen from https://github.com/AllenInstitute/skeleton_plot/blob/main/skeleton_plot/skel_io.py
    # filename = f"{root_id}_{nuc_id}/{root_id}_{nuc_id}.h5"
    '''
    """loads a meshwork file from .h5 into meshparty.meshwork object

    Args:
        directory (str): directory location of meshwork .h5 file. in cloudpath format as seen in https://github.com/seung-lab/cloud-files
        filename (str): full .h5 filename

    Returns:
        meshwork (meshparty.meshwork): meshwork object containing .h5 data
    """'''

    if "://" not in directory:
        directory = "file://" + directory

    cf = CloudFiles(directory)
    binary = cf.get([filename])

    with io.BytesIO(cf.get(binary[0]["path"])) as f:
        f.seek(0)
        mw = meshwork.load_meshwork(f)

    return mw


meshwork_path = "gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons/axon_dendrite_classifier/groundtruth_mws"
meshwork_cf = CloudFiles(meshwork_path)

ground_truth_path = "gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons/axon_dendrite_classifier/groundtruth_feats_and_class"
ground_truth_cf = CloudFiles(ground_truth_path)

# %%


for meshwork_file in meshwork_cf.list():
    if meshwork_file.endswith(".h5"):
        mw = load_mw(meshwork_path, meshwork_file)
        break

# %%

for ground_truth_file in ground_truth_cf.list():
    if ground_truth_file.endswith(".csv"):
        ground_truth = ground_truth_cf.get(ground_truth_file)
        ground_truth_df = pd.read_csv(BytesIO(ground_truth), index_col=0)
        break

# %%


ground_truth_root_ids = [
    int(name.split("_")[0]) for name in ground_truth_cf.list() if name.endswith(".csv")
]
meshwork_root_ids = [
    int(name.split("mesh")[0]) for name in meshwork_cf.list() if name.endswith(".h5")
]
has_ground_truth = np.intersect1d(ground_truth_root_ids, meshwork_root_ids)


def string_to_list(string):
    string = re.sub("\s+", ",", string)
    if string.startswith("[,"):
        string = "[" + string[2:]
    return eval(string)


label_map = {0: "dendrite", 1: "axon"}

all_l2_dfs = []
for root_id in tqdm(has_ground_truth[:]):
    ground_truth_file = f"{root_id}_feats.csv"
    ground_truth_bytes = ground_truth_cf.get(ground_truth_file)
    ground_truth_df = pd.read_csv(BytesIO(ground_truth_bytes), index_col=0)

    meshwork_file = f"{root_id}mesh.h5"
    mw = load_mw(meshwork_path, meshwork_file)

    ground_truth_df["segment"] = ground_truth_df["segment"].apply(string_to_list)

    root_skel_id = list(mw.root_skel)[0]

    ground_truth_df["classification"] = ground_truth_df["classification"].map(label_map)

    has_soma = ground_truth_df["segment"].apply(lambda x: root_skel_id in x)
    ground_truth_df.loc[has_soma, "classification"] = "soma"

    l2_index = pd.Index(mw.anno["lvl2_ids"]["lvl2_id"])
    l2_df = pd.Series(
        data=mw.skeleton.mesh_to_skel_map, index=l2_index, name="skeleton_index"
    ).to_frame()

    skeleton_df = ground_truth_df.explode("segment").set_index("segment")
    skeleton_df.index.name = "skeleton_index"

    l2_df["classification"] = l2_df["skeleton_index"].map(skeleton_df["classification"])
    l2_df["root_id"] = root_id

    all_l2_dfs.append(l2_df)

all_l2_df = pd.concat(all_l2_dfs)

# %%

client = CAVEclient("minnie65_phase3_v1")

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=5)

wrangler.set_manifest(object_ids=all_l2_df["root_id"], level2_ids=all_l2_df.index)
wrangler.query_level2_shape_features().query_level2_synapse_features(method="time")
# %%
wrangler.aggregate_features_by_neighborhood(
    neighborhood_hops=5, aggregations=["mean", "std"]
)

# %%
features = wrangler.features_
# wrangler.features_.to_csv("ground_truth_features.csv")

# %%
label_df = (
    all_l2_df.reset_index()
    .set_index(["root_id", "lvl2_id"])
    .rename(index={"lvl2_id": "l2_id", "root_id": "object_id"})
    .drop(columns="skeleton_index")
)
# label_df.to_csv("ground_truth_classifications.csv")

# %%
X = features.drop(
    columns=[col for col in features.columns if "rep_coord" in col]
).dropna()
# %%

lda = Pipeline(
    [
        ("transformer", QuantileTransformer(output_distribution="normal")),
        ("classifier", LinearDiscriminantAnalysis()),
    ]
)
rf = RandomForestClassifier(n_estimators=500, max_depth=4, n_jobs=-1)

models = {"lda": lda, "rf": rf}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

rows = []
for fold, (train_indices, test_indices) in enumerate(kf.split(has_ground_truth)):
    print("Fold:", fold)
    train_roots = has_ground_truth[train_indices]
    test_roots = has_ground_truth[test_indices]

    X_train = features.loc[train_roots].dropna()
    X_test = features.loc[test_roots].dropna()

    y_train = label_df.loc[X_train.index]["classification"]
    y_test = label_df.loc[X_test.index]["classification"]

    for model_name, model in models.items():
        print("Model:", model_name)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        report = classification_report(y_train, y_train_pred, output_dict=True)
        row = {
            "fold": fold,
            "model": model_name,
            "evaluation": "train",
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
        rows.append(row)

        report = classification_report(y_test, y_test_pred, output_dict=True)
        row = {
            "fold": fold,
            "model": model_name,
            "evaluation": "test",
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
        }
        rows.append(row)

results = pd.DataFrame(rows)

# %%

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

sns.stripplot(data=results, x="model", y="accuracy", hue="evaluation", ax=ax)

# %%
final_rf = RandomForestClassifier(n_estimators=500, max_depth=4, n_jobs=-1)
y = label_df.loc[X.index]["classification"]

final_rf.fit(X, y)
y_pred = final_rf.predict(X)

print(classification_report(y, y_pred))

# %%

joblib.dump(final_rf, "emily_ground_truth_a-d-s.joblib")

# %%

importances = (
    pd.Series(data=final_rf.feature_importances_, index=X.columns, name="importance")
    .to_frame()
    .reset_index()
)

fig, ax = plt.subplots(1, 1, figsize=(6, 10))

sns.barplot(
    data=importances,
    x="importance",
    y="index",
    ax=ax,
)

# %%

temp_path = Path("troglobyte-sandbox/temp_models")
out_path = Path("troglobyte-sandbox/models")

model_name = "local_compartment_classifier_ej_skeletons"
temp_name = temp_path / (model_name + ".skops")
with open(temp_name, mode="bw") as f:
    dump(final_rf, file=f)

if not out_path.exists():
    skops.hub_utils.init(
        model=temp_name,
        requirements=["scikit-learn"],
        dst=out_path / model_name,
        task="tabular-classification",
        data=X,
    )

skops.hub_utils.add_files(__file__, dst=out_path / model_name, exist_ok=True)

with open(out_path / model_name / f"{model_name}.skops", mode="bw") as f:
    dump(final_rf, file=f)


model_card = card.Card(model, metadata=card.metadata_from_config(out_path / model_name))

model_card.metadata.license = "mit"
limitations = "This model is not ready to be used in production."
model_description = (
    "This is a model trained to classify pieces of neuron as axon, dendrite, or soma, "
    "based only on their local shape and synapse features."
    "The model is a random forest classifier which was trained on neuron compartment "
    "labels generated by Emily Joyce for 53 neurons in the Minnie65 Phase3 dataset."
)
model_card_authors = "bdpedigo"
model_card.add(
    model_card_authors=model_card_authors,
    limitations=limitations,
    model_description=model_description,
)

model_card.save(out_path / model_name / "README.md")


skops.hub_utils.push(
    repo_id=f"bdpedigo/{model_name}",
    source=out_path / model_name,
    create_remote=False,
    private=True,
)
