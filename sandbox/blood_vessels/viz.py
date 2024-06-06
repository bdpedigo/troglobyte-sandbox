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
wrangler.query_level2_networks()

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
relevant_features = features[model.feature_names_in_].dropna()
transformed = model.transform(relevant_features)
pred_labels = model.predict(relevant_features)
transformed = pd.DataFrame(
    transformed, columns=["LDA1", "LDA2", "LDA3"], index=relevant_features.index
)

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

# %%
palette = dict(zip(og_labels["label"].unique(), sns.color_palette("tab10")))

fig, axs = plt.subplots(1, 2, figsize=(15, 7.5), sharex=True, sharey=True)
ax = axs[0]
sns.scatterplot(
    data=og_transformed,
    x="LDA1",
    y="LDA2",
    hue=og_pred_labels,
    ax=ax,
    # alpha=0.1,
    s=1,
    linewidth=0,
    legend=False,
    palette=palette,
)
ax.set_title("Training data")
ax.set(xticks=[], yticks=[], xlabel="LDA1", ylabel="LDA2")

class_centroids = og_transformed.groupby(og_pred_labels).mean()
for i, centroid in class_centroids.iterrows():
    ax.text(
        centroid["LDA1"],
        centroid["LDA2"],
        f"{i}",
        fontsize=12,
        ha="center",
        va="center",
        color=palette[i],
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
    )


ax = axs[1]
sns.scatterplot(
    data=transformed,
    x="LDA1",
    y="LDA2",
    ax=ax,
    alpha=0.1,
    s=1,
    linewidth=0,
    hue=pred_labels,
    palette=palette,
    legend=False,
)
ax.set_title("Vasculature data")
ax.set(xticks=[], yticks=[], xlabel="LDA1", ylabel="LDA2")

class_centroids = transformed.groupby(pred_labels).mean()
for i, centroid in class_centroids.iterrows():
    ax.text(
        centroid["LDA1"],
        centroid["LDA2"],
        f"{i}",
        fontsize=12,
        ha="center",
        va="center",
        color=palette[i],
        path_effects=[pe.withStroke(linewidth=3, foreground="black")],
    )
# ax.xaxis.set_label_params(labelbottom=True)
ax.xaxis.set_label_text("LDA1")

ax.axhline(-1.25, color="black", linewidth=0.5)
ax.axhline(-0.25, color="black", linewidth=0.5)
ax.axvline(-0.5, color="black", linewidth=0.5)
ax.axvline(-1.25, color="black", linewidth=0.5)

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
