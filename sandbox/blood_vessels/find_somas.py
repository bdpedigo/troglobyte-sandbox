# %%
from caveclient import CAVEclient

client = CAVEclient("minnie65_public")
client.materialize.version = 1078
cell_types_table = client.materialize.query_table("aibs_metamodel_celltypes_v661")

neuron_types = cell_types_table.query(
    "classification_system.isin(['excitatory_neuron', 'inhibitory_neuron'])"
)
nonneuron_types = cell_types_table.query("classification_system == 'nonneuron'")

# %%
supervoxels = neuron_types["pt_supervoxel_id"]

# %%
level = 4
chunkedgraph_nodes = client.chunkedgraph.get_roots(supervoxels.values, stop_layer=level)

# %%
import pandas as pd

chunkedgraph_nodes = pd.Series(chunkedgraph_nodes, index=neuron_types.index)

# %%

from tqdm import tqdm

sample_nodes = chunkedgraph_nodes.sample(20)

level2s_by_id = []
low_layer = 2
for sample_node in tqdm(sample_nodes):
    level2s_for_root = client.chunkedgraph.get_leaves(sample_node, stop_layer=low_layer)
    level2s = pd.Series(
        data=level2s_for_root,
        index=[sample_node] * len(level2s_for_root),
        name="level2_id",
    )
    level2s.index.name = "level4_id"
    level2s_by_id.append(level2s.to_frame())

level2s_df = pd.concat(level2s_by_id)

from nglui import statebuilder

img = statebuilder.ImageLayerConfig(
    source=client.info.image_source(),
)
seg = statebuilder.SegmentationLayerConfig(
    source=client.info.segmentation_source(),
    fixed_ids=level2s_df["level2_id"].values,
    active=True,
)

sb = statebuilder.StateBuilder(
    layers=[img, seg],
    target_site="mainline",
    view_kws={"zoom_3d": 0.001, "zoom_image": 0.0000001},
)

sb.render_state(return_as="html")

# %%
