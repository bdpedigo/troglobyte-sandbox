# %%
import pickle
from pathlib import Path
from time import sleep

import hvplot.pandas  # noqa
import pandas as pd
import panel as pn
import thisnotthat as tnt
from caveclient import CAVEclient
from nglui import statebuilder
from sklearn.decomposition import PCA
from skops.io import load

from troglobyte.features import CAVEWrangler

client = CAVEclient("minnie65_phase3_v1")

out_path = Path("troglobyte-sandbox/results/vasculature")
wrangler_path = out_path / "wrangler_stash.pkl"

wrangler: CAVEWrangler = pickle.load(open(wrangler_path, "rb"))
wrangler.client = client

# %%

features = pd.read_csv(
    "troglobyte-sandbox/results/vasculature/vasculature_features_box=539648_578560_647680_559104_594944_678400.csv",
    index_col=[0, 1],
)
features = features[features.groupby("object_id").transform("size") > 1]

object_posteriors = (
    features.groupby(["object_id"])[
        [
            "bd_boxes_axon",
            "bd_boxes_dendrite",
            "bd_boxes_glia",
            "bd_boxes_soma",
        ]
    ]
    .mean()
    .dropna()
)


# %%

model = load(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/local_compartment_classifier_bd_boxes.skops"
)

# %%


def rich_transform(features, model):
    relevant_features = features[model.feature_names_in_].dropna()
    transformed = model.transform(relevant_features)
    transformed = pd.DataFrame(
        transformed, columns=["LDA1", "LDA2", "LDA3"], index=relevant_features.index
    )

    pca_embed = PCA(n_components=2).fit_transform(transformed.values)
    transformed["PCALDA1"] = pca_embed[:, 0]
    transformed["PCALDA2"] = pca_embed[:, 1]

    pred_labels = model.predict(relevant_features)
    transformed["pred_label"] = pred_labels

    posteriors = model.predict_proba(relevant_features)
    transformed["max_posterior"] = posteriors.max(axis=1)

    return transformed


transformed = rich_transform(features, model)

# %%

pn.extension()

df = transformed.reset_index()

plot = tnt.BokehPlotPane(
    df[["LDA1", "LDA2"]],
    show_legend=False,
    labels=df["pred_label"],
    width=450,
    height=450,
    marker_size=0.01,
    line_width=0,
)

# data_view = tnt.SimpleDataPane(
#     df,
# )

# data_view.link(plot, selected="selected", bidirectional=True)

# markdown = pn.pane.Markdown("Test.", width=100)
html = pn.pane.HTML("No selection")


def render_ngl_link(event):
    sleep(0.5)
    sbs = []
    dfs = []
    img_layer, seg_layer = statebuilder.helpers.from_client(client)
    selected = event.new
    selected_df = df.iloc[selected]
    if len(selected_df) > 100:
        selected_df = selected_df.sample(100)
    l2_ids = selected_df["level2_id"].values
    ids = client.chunkedgraph.get_roots(l2_ids, stop_layer=4)
    seg_layer.add_selection_map(fixed_ids=ids)

    sb = statebuilder.StateBuilder(layers=[img_layer, seg_layer])
    html.object = sb.render_state(return_as="html")


plot.param.watch(render_ngl_link, "selected")


# labeller = tnt.LabelEditorWidget(plot.labels)
# labeller.link_to_plot(plot)

text_input = pn.widgets.TextInput(
    name="IDs input", placeholder="Enter IDs to label here...", max_length=1_000_000
)


def transfer_selection(event):
    input = event.new
    input = input.split(",")
    selection = []
    for i in input:
        i = i.strip()
        try:
            i = int(i)
            selection.append(i)
        except:
            pass
    plot.selected = selection


text_input.param.watch(transfer_selection, "value")

options = transformed["pred_label"].unique().tolist()
options += ["other"]
select = pn.widgets.Select(name="Select new label", options=options)

button = pn.widgets.Button(name="Apply new label")

col2 = pn.Column(text_input, select, button)
row1 = pn.Row(plot, col2)
row2 = pn.Row(html)
layout = pn.Column(row1, row2)
layout.servable()

# %%
