# %%
import numpy as np
from caveclient import CAVEclient

from troglobyte.features import CAVEWrangler

client = CAVEclient("minnie65_phase3_v1")

# %%
wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=False)

start = np.array([263697, 279652, 21026])
end = np.array([266513, 282468, 21314])
bounding_box = np.stack((start, end))
center = (start + end) // 2

volume_query = wrangler.query_objects_from_box(
    bounding_box, source_resolution=[4, 4, 40], size_threshold=100
)
# %%
query = volume_query.deepcopy()
query.sample_objects(n=250).set_query_box_from_point(
    center, box_width=30_000, source_resolution=[4, 4, 40]
).query_synapse_features().query_level2_shape_features().join_level2_features()
