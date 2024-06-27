# %%
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from caveclient import CAVEclient
from pcg_skel import pcg_skeleton_direct
from tqdm.auto import tqdm

from networkframe import NetworkFrame
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
from skops.io import load

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

    pred_labels = model.predict(relevant_features)
    transformed["pred_label"] = pred_labels

    posteriors = model.predict_proba(relevant_features)
    transformed["max_posterior"] = posteriors.max(axis=1)

    return transformed


transformed = rich_transform(features, model)

# %%
import hvplot.pandas  # noqa

import panel as pn


pn.extension()


fig = transformed.hvplot.scatter(
    x="LDA1",
    y="LDA2",
    c="pred_label",
    cmap="Category10",
    alpha=0.1,
    size=0.5,
    legend="top",
    height=500,
    width=500,
    hover=False,
)
pn.Row(fig, transformed.sample(10)).servable()


# %%
import holoviews as hv
import pandas as pd
from bokeh.plotting import show

hv.extension("bokeh")

macro = hv.Table(
    transformed.reset_index(drop=True).sample(10_000),
    kdims=["LDA1", "LDA2", "LDA3"],
    vdims=["pred_label", "max_posterior"],
)

scatter = macro.to.scatter("LDA1", "LDA2")
#
# hv.output(max_frames=1000)

# show the plot
scatter.opts(
    width=450,
    height=450,
    color="pred_label",
    cmap="Category10",
    size=1,
    tools=["hover"],
    alpha=0.1,
    show_legend=False,
)

show(hv.render(scatter))
# %%
scatter

# %%


# colors = sns.color_palette("pastel", n_colors=5).as_hex()
# for i, (label, ids) in enumerate(object_predictions.groupby(object_predictions)):
#     # ids = ids.sample(max(1, len(ids)))
#     sub_df = ids.to_frame().reset_index()
#     sub_df["color"] = colors[i]
#     new_seg_layer = statebuilder.SegmentationLayerConfig(
#         source=client.info.segmentation_source(),
#         name=label,
#         selected_ids_column=grouping,
#         color_column="color",
#         alpha_3d=0.3,
#     )
#     sbs.append(statebuilder.StateBuilder(layers=[new_seg_layer]))
#     dfs.append(sub_df)

# sb = statebuilder.ChainedStateBuilder(sbs)

# sb.render_state(dfs, return_as="html")


# %%
from time import sleep

import pandas as pd
import panel as pn
import seaborn as sns
import thisnotthat as tnt
from caveclient import CAVEclient
from nglui import statebuilder
from skops.io import load

from troglobyte.features import CAVEWrangler

pn.extension()

df = transformed.reset_index()

plot = tnt.BokehPlotPane(
    df[["LDA1", "LDA3"]],
    show_legend=False,
    labels=df["pred_label"],
    width=450,
    height=450,
    marker_size=0.01,
    line_width=0,
)

data_view = tnt.SimpleDataPane(
    df,
)


data_view.link(plot, selected="selected", bidirectional=True)


# markdown = pn.pane.Markdown("Test.", width=100)
html = pn.pane.HTML("<h1>Test</h1>", width=100)


def update_markdown(event):
    sleep(1)
    html.object = f"Selected: {len(event.new)}"


def render_ngl_link(event):
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
# text_input = pn.widgets.TextInput(value=markdown.object)
# text_input.link(markdown, value="object")
# new_panel.link(data_view)
pn.Row(html, plot, data_view).servable()


# %%
import panel as pn
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

pn.extension()

palette = dict(
    zip(transformed["pred_label"].unique(), sns.color_palette("tab10").as_hex())
)

transformed["mapped_colors"] = transformed["pred_label"].map(palette)
source = ColumnDataSource(transformed.reset_index(drop=True))
p = figure(width=400, height=400)
p.scatter("LDA1", "LDA2", source=source, color="mapped_colors", size=0.1, alpha=0.1)
p.axis.axis_label = None
p.axis.visible = False
p.grid.grid_line_color = None
bokeh_pane = pn.pane.Bokeh(p)

bokeh_pane


# %%

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")

og_features = pd.read_csv(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/features.csv"
)
og_relevant_features = og_features[model.feature_names_in_].dropna()
og_transformed = model.transform(og_relevant_features)
og_transformed = pd.DataFrame(og_transformed, columns=["LDA1", "LDA2", "LDA3"])
og_labels = pd.read_csv(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/labels.csv"
)
og_pred_labels = model.predict(og_relevant_features)

from sklearn.neighbors import LocalOutlierFactor

outlier_model = LocalOutlierFactor(n_neighbors=25, novelty=True)
# outlier_model = IsolationForest(n_estimators=200)
# outlier_model = OneClassSVM(kernel="rbf", nu=0.1)
outlier_model.fit(og_transformed)
og_pred_outliers = outlier_model.predict(og_transformed)

pred_outliers = outlier_model.predict(transformed)


# %%
decision = model.decision_function(og_relevant_features)
decision -= decision.min()
og_decision_max = decision.max(axis=1)

decision = model.decision_function(relevant_features)
decision -= decision.min()
decision_max = decision.max(axis=1)


palette = dict(zip(og_labels["label"].unique(), sns.color_palette("tab10")))
palette["novel"] = "black"

fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex="col", sharey="row")


def single_scatter(data, labels, ax, x, y, outliers=None, decision=None):
    if outliers is not None:
        labels[outliers == -1] = "novel"
    if decision is not None:
        data["size"] = decision / decision.max()
        size = "size"
    else:
        size = None

    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=labels,
        ax=ax,
        # alpha=alpha,
        size=size,
        sizes=(1, 20),
        # alpha=0.1,
        # s=1,
        linewidth=0,
        legend=False,
        palette=palette,
    )

    ax.set(xticks=[], yticks=[], xlabel=x, ylabel=y)

    class_centroids = data.groupby(labels).mean()
    for i, centroid in class_centroids.iterrows():
        if i == "novel":
            continue
        ax.text(
            centroid[x],
            centroid[y],
            f"{i}",
            fontsize=12,
            ha="center",
            va="center",
            color=palette[i],
            path_effects=[pe.withStroke(linewidth=3, foreground="black")],
        )


ax = axs[0, 0]
single_scatter(
    og_transformed,
    og_pred_labels,
    ax,
    "LDA1",
    "LDA2",
    outliers=og_pred_outliers,
    decision=og_decision_max,
)
ax.set_title("Training data")

ax = axs[0, 1]
single_scatter(
    transformed,
    pred_labels,
    ax,
    "LDA1",
    "LDA2",
    outliers=pred_outliers,
    decision=decision_max,
)
ax.set_title("Vasculature data")

ax = axs[1, 0]
single_scatter(
    og_transformed,
    og_pred_labels,
    ax,
    "LDA2",
    "LDA3",
    outliers=og_pred_outliers,
    decision=og_decision_max,
)

ax = axs[1, 1]
single_scatter(
    transformed,
    pred_labels,
    ax,
    "LDA2",
    "LDA3",
    outliers=pred_outliers,
    decision=decision_max,
)

# ax = axs[0]
# sns.scatterplot(
#     data=og_transformed,
#     x="LDA1",
#     y="LDA2",
#     hue=og_pred_labels,
#     ax=ax,
#     # alpha=0.1,
#     s=1,
#     linewidth=0,
#     legend=False,
#     palette=palette,
# )
# ax.set_title("Training data")
# ax.set(xticks=[], yticks=[], xlabel="LDA1", ylabel="LDA2")

# class_centroids = og_transformed.groupby(og_pred_labels).mean()
# for i, centroid in class_centroids.iterrows():
#     ax.text(
#         centroid["LDA1"],
#         centroid["LDA2"],
#         f"{i}",
#         fontsize=12,
#         ha="center",
#         va="center",
#         color=palette[i],
#         path_effects=[pe.withStroke(linewidth=3, foreground="black")],
#     )


# ax = axs[1]
# sns.scatterplot(
#     data=transformed,
#     x="LDA1",
#     y="LDA2",
#     ax=ax,
#     alpha=0.1,
#     s=1,
#     linewidth=0,
#     hue=pred_labels,
#     palette=palette,
#     legend=False,
# )
# ax.set_title("Vasculature data")
# ax.set(xticks=[], yticks=[], xlabel="LDA1", ylabel="LDA2")

# class_centroids = transformed.groupby(pred_labels).mean()
# for i, centroid in class_centroids.iterrows():
#     ax.text(
#         centroid["LDA1"],
#         centroid["LDA2"],
#         f"{i}",
#         fontsize=12,
#         ha="center",
#         va="center",
#         color=palette[i],
#         path_effects=[pe.withStroke(linewidth=3, foreground="black")],
#     )
# # ax.xaxis.set_label_params(labelbottom=True)
# ax.xaxis.set_label_text("LDA1")

# # ax.axhline(-1.25, color="black", linewidth=0.5)
# # ax.axhline(-0.25, color="black", linewidth=0.5)
# # ax.axvline(-0.5, color="black", linewidth=0.5)
# # ax.axvline(-1.25, color="black", linewidth=0.5)
# %%

# contour plot for each class's posterior
fig, axs = plt.subplots(1, 2, figsize=(20, 10))


sns.scatterplot(
    x=transformed["LDA1"],
    y=transformed["LDA2"],
    hue=1 - posteriors.max(axis=1) + 0.1,
    legend=False,
    s=2,
    palette="Greys",
    linewidth=0,
    ax=axs[0],
    hue_norm=(0, 1),
    # legend=False,
    # palette=palette,
    # size=posteriors.max(axis=1),
    # sizes=(0.5, 1),
    # linewidth=0,
)
axs[0].set_title("Uncertainty rating")

novelty_posteriors = outlier_model.decision_function(
    transformed[["LDA1", "LDA2", "LDA3"]]
)
sns.scatterplot(
    x=transformed["LDA1"],
    y=transformed["LDA2"],
    hue=novelty_posteriors.max() - novelty_posteriors,
    legend=False,
    s=2,
    # size=noverl
    palette="Greys",
    linewidth=0,
    ax=axs[1],
    hue_norm=(0, 1),
)
axs[1].set_title("Novelty rating")

# %%


# %%
lda = model.steps[1][1]

# %%
vas_outliers_index = relevant_features.index[pred_outliers == -1]
vas_outliers_transformed = transformed.loc[vas_outliers_index]

# %%
sns.clustermap(vas_outliers_transformed.drop(columns="size"))

# %%
# query_data = transformed.query("LDA1 < 1.5 & LDA1 > 0 & LDA2 > 5.5 & LDA2 < 7.5")
query_data = transformed.query(
    "LDA1 < -0.25 & LDA1 > -1.25 & LDA2 > -1.25 & LDA2 < -0.5"
)
query_data.index.get_level_values("object_id").value_counts().sort_values(
    ascending=False
).index[:20]

# %%
object_posteriors.idxmax(axis=1)

# %%
features[
    [
        "pca_unwrapped_0",
        "pca_unwrapped_1",
        "pca_unwrapped_2",
    ]
]


# %%
skeleton_nfs_path = out_path / "skeleton_nfs.pkl"
if not os.path.exists(skeleton_nfs_path):
    skeleton_nfs = {}
    for object_id, nf in tqdm(
        wrangler.object_level2_networks_.items(),
        total=len(wrangler.object_level2_networks_),
    ):
        if nf.edges.shape[0] >= 5:
            nf = NetworkFrame(nf.nodes.copy(), nf.edges.copy())
            nf.largest_connected_component(inplace=True, directed=False)
            nodes, edges = nf.to_simple_nodes_edges(
                nodes_columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"]
            )
            skel = pcg_skeleton_direct(nodes, edges)
            order_to_skeleton_index = skel.mesh_to_skel_map
            nf.nodes["skeleton_index"] = order_to_skeleton_index
            skeleton_nodes = nf.nodes.groupby("skeleton_index").mean()
            size_nm3 = nf.nodes.groupby("skeleton_index")["size_nm3"].sum()
            skeleton_nodes["size_nm3"] = size_nm3
            level2_nodes_by_skeleton = (
                nf.nodes.reset_index().groupby("skeleton_index")["level2_id"].unique()
            )
            skeleton_nodes["level2_ids"] = skeleton_nodes.index.map(
                level2_nodes_by_skeleton
            )
            skeleton_edges = pd.DataFrame(data=skel.edges, columns=["source", "target"])
            skeleton_nf = NetworkFrame(skeleton_nodes, skeleton_edges)
            skeleton_nf.apply_node_features(
                ["rep_coord_x", "rep_coord_y", "rep_coord_z"], inplace=True
            )
            skeleton_nf.edges["length"] = np.linalg.norm(
                skeleton_nf.edges[
                    ["source_rep_coord_x", "source_rep_coord_y", "source_rep_coord_z"]
                ].values
                - skeleton_nf.edges[
                    ["target_rep_coord_x", "target_rep_coord_y", "target_rep_coord_z"]
                ].values,
                axis=1,
            )
            skeleton_nfs[object_id] = skeleton_nf
            segments = skeleton_nf.k_hop_decomposition(k=5)
            radii = pd.Series(index=segments.index)
            for segment_id, segment in segments.items():
                paths = segment.shortest_paths(directed=False, weight_col="length")
                total_length = paths.max().max()
                total_volume = segment.nodes["size_nm3"].sum()
                radius = np.sqrt(total_volume / (np.pi * total_length))
                radii[segment_id] = radius
            skeleton_nf.nodes["radius"] = radii

    with open(skeleton_nfs_path, "wb") as f:
        pickle.dump(skeleton_nfs, f)

else:
    skeleton_nfs = pickle.load(open(skeleton_nfs_path, "rb"))


# %%

pv.set_jupyter_backend("trame")


import numpy as np

tubes = pv.MultiBlock()

for skeleton_nf in skeleton_nfs.values():
    mean_axon_posterior = skeleton_nf.nodes["bd_boxes_axon_neighbor_mean"].mean()
    if mean_axon_posterior > 0.8:
        nodes, edges = skeleton_nf.to_simple_nodes_edges(
            nodes_columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"]
        )

        # find approximate tangent vectors for each edge
        tangent_vectors_by_edge = np.abs(nodes[edges[:, 1]] - nodes[edges[:, 0]])
        tangent_vectors_by_edge /= np.linalg.norm(tangent_vectors_by_edge, axis=1)[
            :, None
        ]

        tangent_vectors_by_node = np.zeros_like(nodes)
        for i, edge in enumerate(edges):
            tangent_vectors_by_node[edge] += tangent_vectors_by_edge[i]
        tangent_vectors_by_node /= np.abs(
            np.linalg.norm(tangent_vectors_by_node, axis=1)[:, None]
        )

        padded_lines = np.concatenate((np.full((len(edges), 1), 2), edges), axis=1)
        skeleton_poly = pv.PolyData(nodes, lines=padded_lines)
        skeleton_poly["tangent_vectors"] = tangent_vectors_by_edge
        skeleton_poly["radius"] = skeleton_nf.nodes["radius"].values
        min_radius = skeleton_nf.nodes["radius"].min()
        max_radius = skeleton_nf.nodes["radius"].max()

        # node_colors = [colors[i] for i in skeleton_nf.nodes.index]
        # # node_colors = np.tile(node_colors, 20)
        # node_colors = np.repeat(node_colors, 20, axis=0)

        tube = skeleton_poly.tube(
            scalars="radius",
            radius=min_radius,
            absolute=True,
            n_sides=8,
            #   radius_factor=max_radius / min_radius
        )
        # tube["TubeNormals"] = np.abs(tube["TubeNormals"])
        # tube["colors"] = np.repeat(tangent_vectors_by_node, 20, axis=0)
        tubes.append(tube)

tubes = tubes.combine()
plotter = pv.Plotter()
plotter.add_mesh(tubes, scalars="tangent_vectors", rgb=True)
# plotter.add_mesh_clip_plane(tubes, scalars='tangent_vectors', rgb=True)
plotter.enable_fly_to_right_click()
plotter.add_axes(
    line_width=5,
    cone_radius=0.6,
    shaft_length=0.7,
    tip_length=0.3,
    ambient=0.5,
    label_size=(0.4, 0.16),
)
plotter.export_html("tubes.html")
plotter.show()

# %%
object_posteriors.idxmax(axis=1).value_counts()
