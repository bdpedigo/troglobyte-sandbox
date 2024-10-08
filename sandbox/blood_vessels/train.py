# %%
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from tqdm.autonotebook import tqdm

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

# %%

data_path = Path("troglobyte-sandbox/data/blood_vessels")

new_segments = pd.read_csv(
    data_path / "bdp_perivasculature_clean_ids.csv", header=None
).values.ravel()
new_segments = np.unique(new_segments)

# %%
feature_path = Path("troglobyte-sandbox/results/vasculature")


new_feature_df = []
for filename in tqdm(glob.glob(str(feature_path / "vasculature_features_*.csv"))):
    sub_feature_df = pd.read_csv(filename, index_col=[0, 1])
    sub_feature_df = sub_feature_df[feature_df.columns]
    sub_feature_df = sub_feature_df.loc[
        sub_feature_df.index.get_level_values("object_id").intersection(new_segments)
    ]
    new_feature_df.append(sub_feature_df)

new_feature_df = pd.concat(new_feature_df)
new_feature_df = new_feature_df.dropna()

# %%
new_feature_df = new_feature_df.loc[new_feature_df.index.drop_duplicates()]

# %%
new_labels = pd.Series(
    data="perivascular",
    index=new_feature_df.index.get_level_values("object_id").unique(),
    name="label",
)

# %%
combined_feature_df = pd.concat([feature_df, new_feature_df])
combined_labels = pd.concat([labels, new_labels])
combined_labels_by_level2 = combined_feature_df.index.get_level_values("object_id").map(
    combined_labels
)
combined_labels_by_level2 = combined_labels_by_level2.to_series(
    index=combined_feature_df.index
)

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

from sklearn.mixture import GaussianMixture

n_components_by_label = {
    "perivascular": 2,
    "dendrite": 3,
    "axon": 2,
    "soma": 2,
    "glia": 5,
}

for label, sub_X_transformed_df in X_transformed_df.groupby("label"):
    sub_X_transformed_df = sub_X_transformed_df.drop(columns="label")[
        [f"LDA{i}" for i in range(3)]
    ].copy()
    gmm = GaussianMixture(n_components=n_components_by_label[label])
    gmm.fit(sub_X_transformed_df)
    sub_X_transformed_df["gmm_label"] = gmm.predict(sub_X_transformed_df)
    pg = sns.PairGrid(
        sub_X_transformed_df, hue="gmm_label", corner=True, palette="tab10"
    )
    pg.map_lower(sns.scatterplot, alpha=0.1, linewidth=0, s=1)
    pg.map_diag(sns.histplot, kde=True)
    pg.set(title=label)

# %%

for filename in tqdm(glob.glob(str(feature_path / "vasculature_features_*.csv"))[2:]):
    vasculature_feature_df = pd.read_csv(filename, index_col=[0, 1])
    vasculature_feature_df = vasculature_feature_df[feature_df.columns]
    break

# %%
vasculature_feature_df = vasculature_feature_df.dropna()[model.feature_names_in_]

# %%
vasculature_pred = model.predict(vasculature_feature_df)

vasculature_X_transformed = model.transform(vasculature_feature_df)

# %%
vasculature_X_transformed_df = pd.DataFrame(
    data=vasculature_X_transformed, index=vasculature_feature_df.index
)
vasculature_X_transformed_df.columns = [
    f"LDA{i}" for i in range(vasculature_X_transformed_df.shape[1])
]
vasculature_X_transformed_df["label"] = vasculature_pred

pg = sns.PairGrid(vasculature_X_transformed_df, hue="label", corner=True)
pg.map_lower(sns.scatterplot, alpha=0.1, linewidth=0, s=1)
pg.map_diag(sns.histplot, kde=True)
pg.add_legend(markerscale=10)

# %%

vasculature_pred_df = pd.DataFrame(
    data=vasculature_pred, index=vasculature_feature_df.index, columns=["label"]
)

# %%
level2_counts_by_root = vasculature_pred_df.groupby("object_id").size()
big_roots = level2_counts_by_root[level2_counts_by_root > 5].index

# %%
big_vasculature_pred_df = vasculature_pred_df.loc[big_roots]


# %%
sub_vasculature_index = (
    big_vasculature_pred_df.index.get_level_values("object_id")
    .unique()
    .to_series()
    .sample(100)
)
sub_vasculature_pred_df = vasculature_pred_df.loc[sub_vasculature_index]


# %%
level2_ids = sub_vasculature_pred_df.index.get_level_values("level2_id")

# %%
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1")
client.materialize.version = 1078
timestamp = client.materialize.get_timestamp(client.materialize.version)

# %%

max_level = 6
for level in range(3, max_level + 1):
    level_ids = client.chunkedgraph.get_roots(
        level2_ids, stop_layer=level, timestamp=timestamp
    )
    sub_vasculature_pred_df[f"level{level}_id"] = level_ids

# %%
sub_vasculature_pred_df = sub_vasculature_pred_df.reset_index()


# %%
current_level = max_level
current_df = sub_vasculature_pred_df

labels_by_level = []

for current_level in range(max_level, 1, -1):
    label_counts_at_level = current_df.groupby([f"level{current_level}_id"])[
        "label"
    ].nunique()
    singletons = label_counts_at_level[label_counts_at_level == 1].index
    new_labels = (
        current_df.set_index(f"level{current_level}_id")
        .loc[singletons]
        .groupby([f"level{current_level}_id"])["label"]
        .first()
    )
    new_labels = new_labels.to_frame()
    new_labels["level"] = current_level
    new_labels.index = new_labels.index.astype("uint64")
    labels_by_level.append(new_labels)
    current_df = current_df.set_index(f"level{current_level}_id").drop(singletons)
# %%
labels_by_level = pd.concat(labels_by_level)
labels_by_level.index.name = "node_id"
# %%

import numpy as np
from nglui.segmentprops import SegmentProperties

seg_df = labels_by_level.copy()

n_randoms = 10
for i in range(n_randoms):
    seg_df[f"random_{i}"] = np.random.uniform(0, 1, size=len(seg_df))

seg_prop = SegmentProperties.from_dataframe(
    seg_df.reset_index(),
    id_col="node_id",
    label_col="label",
    tag_value_cols="label",
    number_cols=["level"],
)

prop_id = client.state.upload_property_json(seg_prop.to_dict())
prop_url = client.state.build_neuroglancer_url(
    prop_id, format_properties=True, target_site="mainline"
)

from nglui import statebuilder

img = statebuilder.ImageLayerConfig(
    source=client.info.image_source(),
)
seg = statebuilder.SegmentationLayerConfig(
    source=client.info.segmentation_source(),
    segment_properties=prop_url,
    fixed_ids=seg_df.index[0],
    active=True,
)

sb = statebuilder.StateBuilder(
    layers=[img, seg],
    target_site="mainline",
    view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001},
)

sb.render_state()

# %%

ids = [
    162985419217044149,
    163196800260374927,
    163267031565598912,
    163267100285076239,
    163267169004553018,
    163267237724029153,
    163337400309776551,
    163337400309776567,
    163337469029253869,
    163337537748730674,
    163407769053954275,
    163407769053954290,
    163407837773431762,
    163407906492908503,
    163478137798131876,
    163478206517608781,
    163478275237085580,
    163690618420199434,
    163760987164377090,
    163761055883853826,
    163831149750124547,
    163831218469601281,
    163831287189078018,
    163831355908554754,
    163901587213778947,
    163971955957956609,
    163972024677433346,
    235195433322811351,
    236040134204590818,
    236041233716219942,
    237027770524241736,
    306851709374713322,
    307702727079640589,
    307980907816431463,
    308262382793142752,
    308263482304772456,
    308540559234970295,
    308540559234975624,
    308541658746599860,
    308542758258231517,
    308542758258231518,
    308542758258232821,
    308822034211687673,
    308823133723311366,
    308824233234937849,
    308824233234937862,
    309110106258161777,
    309110106258164001,
    309111205769790311,
    309391581234873187,
    309392680746498182,
    309392680746498192,
    378953279582837967,
    379516229536253590,
    380637731396619090,
    380642129443093516,
    522699031753504265,
    524950831567137697,
]

level_by_id = labels_by_level.loc[ids, "level"]

# %%
ids = []
for node_id, level in level_by_id.items():
    if level == 3:
        ids.append(node_id)
    else:
        row = sub_vasculature_pred_df.set_index(f"level{level}_id").loc[[node_id]]
        ids.extend(row["level3_id"].to_list())

# %%
pd.Series(ids).to_clipboard(index=False)

# %%
