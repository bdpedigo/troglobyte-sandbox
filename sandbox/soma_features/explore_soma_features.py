# %%
from pathlib import Path

import pandas as pd

data_path = Path(
    "troglobyte-sandbox/data/soma_features/microns_SomaData_AllCells_v661.csv"
)

feature_df = pd.read_csv(data_path, index_col=0, engine="python")

# %%
feature_df = feature_df.drop(
    columns=[
        "cortical_column_labels",
        "predicted_class",
        "predicted_subclass",
        "umap_embedding_x",
        "umap_embedding_y",
        "is_column",
    ]
).set_index(["soma_id", "nucleus_id"])

# %%
import numpy as np
from caveclient import CAVEclient

client = CAVEclient("minnie65_public", version=661)
nuc_table = client.materialize.query_table("nucleus_detection_v0", split_positions=True)
nuc_table = nuc_table.set_index("id")
nuc_table = nuc_table.loc[feature_df.index.get_level_values("nucleus_id")]
nuc_table.index = feature_df.index
positions = nuc_table[["pt_position_x", "pt_position_y", "pt_position_z"]].values

resolution = np.array([4, 4, 40])

positions = positions * resolution

# %%

import pyvista as pv

pv.set_jupyter_backend("client")

plotter = pv.Plotter()
nuc_points = pv.PolyData(positions.astype(float))
plotter.add_mesh(nuc_points, point_size=1, color="red", render_points_as_spheres=True)

plotter.show()

# %%
X = feature_df.copy()
X["pt_position_y"] = nuc_table["pt_position_y"] * 4

X = X.drop(columns=["soma_depth_x", "soma_depth_z"])

# %%

Y = (
    nuc_table[["pt_position_x", "pt_position_z"]]
    .copy()
    .rename(columns={"pt_position_x": "x", "pt_position_z": "z"})
)
Y["x"] = Y["x"] * 4
Y["z"] = Y["z"] * 40
Y.index = X.index
# %%

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

train_index, test_index = train_test_split(X.index, test_size=0.2)

X_train = X.loc[train_index]
X_test = X.loc[test_index]

Y_train = Y.loc[train_index]
Y_test = Y.loc[test_index]

model = RandomForestRegressor(n_estimators=200, n_jobs=-1)

model.fit(X_train, Y_train)

# %%
Y_pred = model.predict(X_test)

# %%
Y_pred_full = pd.DataFrame(Y_pred, index=X_test.index, columns=Y_test.columns)
Y_pred_full["y"] = X_test["pt_position_y"]

Y_test_full = Y_test.copy()
Y_test_full["y"] = X_test["pt_position_y"]

# %%

plotter = pv.Plotter()
nuc_points_pred = pv.PolyData(Y_pred_full.values.astype(float))
nuc_points_true = pv.PolyData(Y_test_full.values.astype(float))

point_size = 5
plotter.add_mesh(
    nuc_points_pred, point_size=point_size, color="red", render_points_as_spheres=True
)
plotter.add_mesh(
    nuc_points_true, point_size=point_size, color="blue", render_points_as_spheres=True
)

plotter.show()


# %%
from sklearn.metrics.pairwise import paired_euclidean_distances

distances = paired_euclidean_distances(Y_test_full, Y_pred_full)


# %%
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution="normal")

X_trans = qt.fit_transform(X)

from sklearn.decomposition import PCA

n_components = 5
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X)

X_pca = pd.DataFrame(
    X_pca, index=X.index, columns=[f"PC{i}" for i in range(1, n_components + 1)]
)

import seaborn as sns

sns.scatterplot(data=X_pca, x="PC1", y="PC2", s=1, linewidth=0, alpha=0.2)

# %%
pg = sns.PairGrid(X_pca)
pg.map_diag(sns.histplot)
pg.map_offdiag(sns.scatterplot, s=1, linewidth=0, alpha=0.05)
