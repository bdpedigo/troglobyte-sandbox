# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from caveclient import CAVEclient
from fiberorient.odf import ODF
from fiberorient.util import make_sphere
from joblib import Parallel, delayed
from matplotlib.colors import rgb2hex
from sklearn.cluster import SpectralClustering
from skops.io import load

from troglobyte.features import CAVEWrangler

pv.set_jupyter_backend("client")

# %%


bd_boxes_model_path = "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/local_compartment_classifier_bd_boxes.skops"
ej_skeleton_model_path = "troglobyte-sandbox/models/local_compartment_classifier_ej_skeletons/local_compartment_classifier_ej_skeletons.skops"


bd_boxes_model = load(bd_boxes_model_path)
ej_skeleton_model = load(ej_skeleton_model_path)


# %%
point_4_4_40 = np.array([124444, 231788, 24950])

client = CAVEclient("minnie65_phase3_v1")

demo = False
if demo:
    currtime = time.time()

    wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=5)
    wrangler.set_query_box_from_point(
        point_4_4_40, box_width=10_000, source_resolution=[4, 4, 40]
    )
    wrangler.query_objects_from_box(size_threshold=20, mip=5)
    wrangler.query_level2_shape_features()
    wrangler.prune_query_to_box()
    wrangler.query_level2_synapse_features(method="existing")

    wrangler.register_model(bd_boxes_model, "bd_boxes")
    wrangler.register_model(ej_skeleton_model, "ej_skeletons")

    wrangler.aggregate_features_by_neighborhood(
        neighborhood_hops=5, aggregations=["mean", "std"]
    )
    wrangler.stack_model_predict_proba("bd_boxes")
    wrangler.stack_model_predict_proba("ej_skeletons")
    print(f"{time.time() - currtime:.3f} seconds elapsed.")


# %%

if demo:
    is_axon_mask = wrangler.features_["bd_boxes_predict_proba_dendrite"] > 0.8

    axon_features = wrangler.features_[is_axon_mask].copy()

    # each of these is a 3D vector
    X = (
        axon_features[
            [
                "pca_unwrapped_0",
                "pca_unwrapped_1",
                "pca_unwrapped_2",
            ]
        ]
        .dropna()
        .values
    )
    X = np.concatenate((X, -X), axis=0)

    # convert to spherical coordinates
    r = np.linalg.norm(X, axis=1)
    theta = np.arccos(X[:, 2] / r)
    phi = np.arctan2(X[:, 1], X[:, 0])

    sns.set_context("talk")

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    sns.scatterplot(x=phi, y=theta, s=5, alpha=0.5)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")


# %%

if demo:
    similarity = np.abs(X @ X.T)

    k = 5
    sc = SpectralClustering(n_clusters=k, affinity="precomputed", n_neighbors=20)

    labels = sc.fit_predict(similarity)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    sns.scatterplot(x=phi, y=theta, hue=labels.astype(str), s=5, alpha=0.5)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")

    axon_features = axon_features.copy()
    axon_features["label"] = labels[: len(labels) // 2]

    n = None
    # sample_features = axon_features.query("label == 1")
    sample_features = axon_features.copy()
    sample_objects = pd.Series(
        sample_features.index.get_level_values("object_id").unique()
    ).sample(n=n)
    sample_features = sample_features.query("object_id in @sample_objects")
    sample_features["R"] = np.abs(sample_features["pca_unwrapped_0"])
    sample_features["G"] = np.abs(sample_features["pca_unwrapped_1"])
    sample_features["B"] = np.abs(sample_features["pca_unwrapped_2"])
    sample_features["hex"] = sample_features[["R", "G", "B"]].apply(
        lambda x: rgb2hex(x), axis=1
    )

    wrangler.visualize_query(
        ids=sample_features.index.get_level_values("level2_id"),
        colors=sample_features["hex"].values,
    )


# %%
if demo:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    sns.histplot(x=phi, y=theta, cbar=True, cmap="Blues", bins=30, ax=ax)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\theta$")
    ax.set_ylim(0, 0.5 * np.pi)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, 0, np.pi])
    ax.set_xticklabels(["-180", "0", "180"])
    ax.set_yticks([0, 0.5 * np.pi])
    ax.set_yticklabels(["0", "90"])

# %%
if demo:
    r = np.linalg.norm(X, axis=1)
    theta = np.arccos(X[:, 2] / r)
    phi = np.arctan2(X[:, 1], X[:, 0])

    counts, phi_edges, theta_edges = np.histogram2d(phi, theta, bins=20)

    theta_edges = np.degrees(theta_edges)  # goes 0 to 180
    phi_edges = np.degrees(phi_edges)  # goes -180 to 180

    plotter = pv.Plotter()
    grid = pv.grid_from_sph_coords(phi_edges, theta_edges, r=0.99)
    grid.cell_data["counts"] = counts.ravel()
    plotter.add_mesh(grid, opacity=1, cmap="Blues")
    points = pv.PointSet(X)
    plotter.add_mesh(points, point_size=3)
    plotter.show()

# %%

# now tile this analysis across an x-z plane

base_box = np.array([[492776, 922152, 993000], [502776, 932152, 1003000]])
x_extent = base_box[1, 0] - base_box[0, 0]
z_extent = base_box[1, 2] - base_box[0, 2]

n_x = 5
n_z = 5

boxes = []
box_names = []
for x_index in range(n_x):
    for z_index in range(n_z):
        box = base_box.copy()
        box[0, 0] += x_index * x_extent
        box[1, 0] += x_index * x_extent
        box[0, 2] -= z_index * z_extent
        box[1, 2] -= z_index * z_extent
        boxes.append(box)
        box_names.append("_".join(map(str, box.flatten())))

OUT_PATH = Path("troglobyte-sandbox/results/axon_direction")

recompute = False
if recompute:

    def extract_features_for_box(box, retry=3):
        try:
            wrangler = CAVEWrangler(client=client, n_jobs=1)
            wrangler.set_query_box(box)
            wrangler.query_objects_from_box(size_threshold=20, mip=5)
            wrangler.query_level2_shape_features()
            wrangler.prune_query_to_box()
            wrangler.query_level2_synapse_features(method="existing")

            wrangler.register_model(bd_boxes_model, "bd_boxes")
            wrangler.register_model(ej_skeleton_model, "ej_skeletons")

            wrangler.aggregate_features_by_neighborhood(
                neighborhood_hops=5, aggregations=["mean", "std"]
            )
            wrangler.stack_model_predict_proba("bd_boxes")
            wrangler.stack_model_predict_proba("ej_skeletons")
            bbox_name = "_".join(map(str, box.flatten()))
            wrangler.features_.to_csv(f"{bbox_name}_features.csv")
            return wrangler
        except:
            if retry > 0:
                return None
            else:
                time.sleep(10)
                return extract_features_for_box(box, retry=retry - 1)

    wranglers = Parallel(n_jobs=-1, verbose=5)(
        delayed(extract_features_for_box)(box) for box in boxes
    )

# %%

box_features = []
for box_name in box_names:
    box_features.append(
        pd.read_csv(OUT_PATH / f"{box_name}_features.csv", index_col=[0, 1])
    )

# %%

# each of these is a 3D vector


sns.set_context("talk")

fig, axs = plt.subplots(n_x, n_z, figsize=(30, 30), sharex=True, sharey=True)

for x_index in range(n_x):
    for z_index in range(n_z):
        ax = axs[x_index, z_index]
        # wrangler = wranglers[x_index * n_z + z_index]
        features = box_features[x_index * n_z + z_index]
        if features is not None:
            is_axon_mask = features["bd_boxes_predict_proba_axon"] > 0.8

            axon_features = features[is_axon_mask].copy()

            X = (
                axon_features[
                    [
                        "pca_unwrapped_0",
                        "pca_unwrapped_1",
                        "pca_unwrapped_2",
                    ]
                ]
                .dropna()
                .values
            )
            X = np.concatenate((X, -X), axis=0)

            # convert to spherical coordinates
            r = np.linalg.norm(X, axis=1)
            theta = np.arccos(X[:, 2] / r)
            phi = np.arctan2(X[:, 1], X[:, 0])

            # sns.scatterplot(x=phi, y=theta, s=5, alpha=0.5)
            # ax.set_xlabel(r"$\phi$")
            # ax.set_ylabel(r"$\theta$")

            sns.histplot(x=phi, y=theta, cbar=True, cmap="Blues", bins=30, ax=ax)
            ax.set_xlabel(r"$\phi$")
            ax.set_ylabel(r"$\theta$")
            ax.set_ylim(0, np.pi)
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xticks([-np.pi, 0, np.pi])
            ax.set_xticklabels(["-180", "0", "180"])
            ax.set_yticks([0, 0.5 * np.pi])
            ax.set_yticklabels(["0", "90"])

# %%
plotter = pv.Plotter(shape=(n_x, n_z))

for x_index in range(n_x):
    for z_index in range(n_z):
        plotter.subplot(x_index, z_index)
        features = box_features[x_index * n_z + z_index]

        if features is not None:
            is_axon_mask = features["bd_boxes_predict_proba_axon"] > 0.8

            axon_features = features[is_axon_mask].copy()

            X = (
                axon_features[
                    [
                        "pca_unwrapped_0",
                        "pca_unwrapped_1",
                        "pca_unwrapped_2",
                    ]
                ]
                .dropna()
                .values
            )
            X = np.concatenate((X, -X), axis=0)

            # convert to spherical coordinates
            r = np.linalg.norm(X, axis=1)
            theta = np.arccos(X[:, 2] / r)
            phi = np.arctan2(X[:, 1], X[:, 0])

            counts, phi_edges, theta_edges = np.histogram2d(
                phi, theta, bins=20, range=[[-np.pi, np.pi], [0, np.pi]]
            )

            theta_edges = np.degrees(theta_edges)  # goes 0 to 180
            phi_edges = np.degrees(phi_edges)  # goes -180 to 180

            grid = pv.grid_from_sph_coords(phi_edges, theta_edges, r=0.99)
            # grid.cell_data["counts"] = counts.ravel()
            plotter.add_mesh(grid, opacity=1, cmap="Blues")
            points = pv.PointSet(X)
            plotter.add_mesh(points, point_size=3)

plotter.link_views()
plotter.show()

# %%

plotter = pv.Plotter(shape=(n_x, n_z))

for x_index in range(n_x):
    for z_index in range(n_z):
        plotter.subplot(x_index, z_index)
        features = box_features[x_index * n_z + z_index]

        if features is not None:
            is_axon_mask = features["bd_boxes_predict_proba_axon"] > 0.8

            axon_features = features[is_axon_mask].copy()

            X = (
                axon_features[
                    [
                        "pca_unwrapped_0",
                        "pca_unwrapped_1",
                        "pca_unwrapped_2",
                    ]
                ]
                .dropna()
                .values
            )
            X = np.concatenate((X, -X), axis=0)

            odf = ODF(degree=8)
            odf.fit(X)

            sphere = make_sphere(3000)
            odf_on_sphere = odf.to_sphere(sphere)
            odf_on_sphere = odf_on_sphere * 2

            faces = sphere.faces.copy()
            new_faces = np.concatenate(
                (np.full(faces.shape[0], 3).reshape(-1, 1), faces), axis=1
            )
            odf_mesh = pv.PolyData(
                sphere.vertices * odf_on_sphere[:, None], faces=new_faces
            )
            odf_mesh["odf"] = odf_on_sphere
            plotter.add_mesh(odf_mesh, scalars="odf")

            # this adds points for the raw data
            points = pv.PointSet(X)
            plotter.add_mesh(points, point_size=1)

            # this adds lines from the origin for the raw data
            # new_X = X.copy()
            # new_X = np.concatenate((np.array([0, 0, 0]).reshape(1, 3), new_X), axis=0)
            # lines = np.concatenate(
            #     [
            #         np.full(len(X), 2).reshape(-1, 1),
            #         np.full(len(X), 0).reshape(-1, 1),
            #         np.arange(1, len(X) + 1)[:, None],
            #     ],
            #     axis=1,
            # )
            # plotter.add_mesh(
            #     pv.PolyData(new_X, lines=lines),
            #     color="black",
            #     line_width=0.1,
            # )

plotter.link_views()
plotter.show()

# %%
