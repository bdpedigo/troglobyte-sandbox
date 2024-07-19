# %%
import pickle
from pathlib import Path

from caveclient import CAVEclient

from troglobyte.features import CAVEWrangler

client = CAVEclient("minnie65_phase3_v1")
client.materialize.version = 1078

out_path = Path("troglobyte-sandbox/results/vasculature")
wrangler_path = out_path / "wrangler_box=745472_587776_774400_762880_679936_819200.pkl"

wrangler: CAVEWrangler = pickle.load(open(wrangler_path, "rb"))
wrangler.client = client

# %%
import pandas as pd

labels = ["dendrite", "axon", "soma", "glia"]
# posterior_features = [f"bd_boxes_{label}_neighbor_mean" for label in labels]
posterior_features = [f"bd_boxes_predict_proba_{label}" for label in labels]
posteriors_by_l2 = wrangler.features_[posterior_features].dropna().copy()

classification_by_l2 = posteriors_by_l2.idxmax(axis=1)

classification_mask = pd.get_dummies(classification_by_l2).astype(int)

# %%
posteriors_by_l2.groupby("object_id").mean()

# %%
object_classification_counts = classification_mask.groupby("object_id").sum()

object_classification_counts["n_level2_ids"] = object_classification_counts.sum(axis=1)

large_object_classification_counts = object_classification_counts.query(
    "n_level2_ids > 5"
).copy()

large_object_classification_counts["max_vote"] = (
    large_object_classification_counts.drop(columns="n_level2_ids").idxmax(axis=1)
)
large_object_classification_counts["max_vote"] = (
    large_object_classification_counts["max_vote"].str.split("_").str[-1]
)
# %%
import numpy as np
from nglui.segmentprops import SegmentProperties

seg_df = large_object_classification_counts.copy()


n_randoms = 10
for i in range(n_randoms):
    seg_df[f"random_{i}"] = np.random.uniform(0, 1, size=len(seg_df))


number_cols = seg_df.columns.to_list()
number_cols.remove("max_vote")
seg_prop = SegmentProperties.from_dataframe(
    seg_df.reset_index(),
    id_col="object_id",
    label_col="max_vote",
    tag_value_cols="max_vote",
    number_cols=number_cols,
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
)

sb = statebuilder.StateBuilder(
    layers=[img, seg], target_site="mainline", view_kws={"zoom_3d": 0.001}
)
sb.render_state()

# %%
index = pd.Index([0, 1])


potential_list = [1]

if potential_list:
    print("True")
else:
    print("False")

# %%
