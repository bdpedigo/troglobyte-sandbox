# %%

from pathlib import Path

import caveclient as cc
import numpy as np
from cloudvolume import Bbox
from joblib import load

from troglobyte.features import L2AggregateWrangler

client = cc.CAVEclient("minnie65_phase3_v1")

model_path = Path("troglobyte-sandbox/models")
model_name = "bethanny_box_model=quantile_lda_train=1-2-3.joblib"

model = load(model_path / model_name)

# %%
cv = client.info.segmentation_cloudvolume()

mip_resolution = cv.mip_resolution(0)

start = np.array([263697, 279652, 21026])
end = np.array([266513, 282468, 21314])
start = start / np.array([2, 2, 1])
end = end / np.array([2, 2, 1])
print(start)
print(end)

bbox = Bbox(start, end, dtype="int32")
unique_labels = cv.unique(bbox)
unique_labels = unique_labels - {0}
unique_labels = np.array(list(unique_labels))
roots = client.chunkedgraph.get_roots(unique_labels)
roots = np.unique(roots)
if roots[0] == 0:
    roots = roots[1:]


# %%

center = (start + end) // 2  # this is in 8 x 8 x 40nm
center_nm = center * np.array([8, 8, 40])
points = [center_nm for _ in roots]

# %%
wrangler = L2AggregateWrangler(
    client,
    verbose=10,
    n_jobs=-1,
    neighborhood_hops=10,
    box_width=10_000,
)
raw_X = wrangler.get_shape_features(roots, points)


# %%
X = raw_X[raw_X.notna().all(axis=1)].copy()
X = X.drop(
    columns=[
        "rep_coord_x",
        "rep_coord_x_neighbor_agg",
        "rep_coord_y",
        "rep_coord_y_neighbor_agg",
        "rep_coord_z",
        "rep_coord_z_neighbor_agg",
    ]
)
# %%


X_transformed = model.transform(X)
