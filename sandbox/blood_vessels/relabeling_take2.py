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
# features = features[features.groupby("object_id").transform("size") > 1]

og_features = pd.read_csv(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/features.csv",
    index_col=[0, 1],
)
og_labels = pd.read_csv(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/labels.csv",
    index_col=[1],
)["label"]

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
og_transformed = rich_transform(og_features, model)

features = pd.concat([features, og_features])

transformed["dataset"] = "new"
og_transformed["dataset"] = "training"
transformed = pd.concat([transformed, og_transformed])

transformed["manual_label"] = transformed.index.get_level_values("object_id").map(
    og_labels
)


# %%

pn.extension()

df = transformed

x = "LDA1"
y = "LDA2"
label = "pred_label"
size = 0.01
plot = tnt.BokehPlotPane(
    df[[x, y]],
    show_legend=False,
    labels=df[label],
    width=450,
    height=450,
    marker_size=size,
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
    # l2_ids = selected_df["level2_id"].values
    l2_ids = selected_df.index.get_level_values("level2_id").values
    # ids = client.chunkedgraph.get_roots(l2_ids, stop_layer=4)
    seg_layer.add_selection_map(fixed_ids=l2_ids)

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
options += ["small"]
select = pn.widgets.Select(name="Select new label", options=options)


def record_labels(event):
    new_label = select.value
    selected = plot.selected
    selected = df.index[selected]
    df.loc[selected, "manual_label"] = new_label
    print(df["manual_label"].notna().sum())


relabel_button = pn.widgets.Button(name="Apply new label")
relabel_button.on_click(record_labels)


def retrain_model(event):
    print("Retraining model...")
    retrain_button.name = "Retraining..."
    new_labels = df["manual_label"]
    new_labels = new_labels.dropna()
    new_features = features.loc[new_labels.index]
    new_features = new_features[model.feature_names_in_].dropna()
    new_labels = new_labels.loc[new_features.index]
    model.fit(new_features, new_labels)
    retrain_button.name = "Retrain model"
    print("Model retrained.")

    # transformed = rich_transform(new_features, model)
    # new_df = transformed[[x, y, label, label]]
    # new_df["size"] = size
    # new_df.columns = plot.dataframe.columns
    # plot.dataframe = new_df
    plot.dataframe['x'] = 0


retrain_button = pn.widgets.Button(name="Retrain model")
retrain_button.on_click(retrain_model)

col2 = pn.Column(text_input, select, relabel_button, retrain_button)
row1 = pn.Row(plot, col2)
row2 = pn.Row(html)
layout = pn.Column(row1, row2)
layout.servable()

# %%
