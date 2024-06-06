# %%

from caveclient import CAVEclient
from skops.io import load

from troglobyte.features import CAVEWrangler

client = CAVEclient("minnie65_phase3_v1")
client.materialize.version = 943

wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=3)

# %%
astrocyte_v943 = [
    864691135081875959,
    864691135308181062,
    864691135568874630,
    864691135576420382,
    864691135609963783,
    864691135657774210,
    864691135737686129,
    864691135776185261,
    864691135938587701,
    864691136488133394,
    864691136577820692,
    864691136904347058,
]


model = load(
    "troglobyte-sandbox/models/local_compartment_classifier_bd_boxes/local_compartment_classifier_bd_boxes.skops"
)

wrangler.set_objects(astrocyte_v943)
wrangler.query_level2_ids()
wrangler.query_level2_shape_features()
wrangler.query_level2_synapse_features(method="existing")
wrangler.query_level2_edges()
wrangler.query_level2_networks(validate=False)
wrangler.register_model(model, "bd_boxes")
wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=5
)

# %%
wrangler.features_.to_csv(
    "troglobyte-sandbox/results/astrocytes/astrocyte_features.csv"
)

# %%

# %%
import seaborn as sns
from caveclient import CAVEclient
from nglui import statebuilder
from skops.io import load

from troglobyte.features import CAVEWrangler

axon_features = wrangler.features_.query("bd_boxes_axon_neighbor_mean > 0.3")

id_list = axon_features.index.get_level_values("level2_id").to_list()
id_list += astrocyte_v943
id_colors = ["#FF0000"] * len(axon_features) + ["#00FF00"] * len(astrocyte_v943)
sbs = []
dfs = []
img_layer, seg_layer = statebuilder.helpers.from_client(client)
seg_layer._view_kws["alpha_3d"] = 0.3
seg_layer.add_selection_map(fixed_ids=id_list, fixed_id_colors=id_colors)
sb = statebuilder.StateBuilder(layers=[img_layer, seg_layer])
temp_state = sb.render_state(return_as="dict")

# Manually add JSON state server
temp_state["jsonStateServer"] = "https://global.daf-apis.com/nglstate/api/v1/post"

current_source = temp_state['layers'][1]['source']

# current_source[:11] + "middleauth+" + current_source[11:]

temp_state['layers'][1]['source'] = current_source[:11] + "middleauth+" + current_source[11:]

# nav_dict = temp_state['navigation']
# nav_dict['pose'] 
# temp_state['navigation']['pose']['position']['voxelSize'] = [4,4,40]

new_sb = statebuilder.StateBuilder(base_state=temp_state)

new_sb.render_state(
    url_prefix="https://spelunker.cave-explorer.org/",
    target_site="cave-explorer",
    return_as="html",
)



# %%
from nglui.statebuilder import StateBuilder

# Build states on base state
new_sb = StateBuilder(
    layers=[seg_layer, origin_layer, target_layer], base_state=base_state
)

# Render as dict
state_dict = new_sb.render_state(
    {"origin": origin_df, "target": target_df},
    return_as="dict",
    url_prefix="https://spelunker.cave-explorer.org/",
    target_site="cave-explorer",
)

# Feed dict to link shortener
state_id = client.state.upload_state_json(state_dict)
url = client.state.build_neuroglancer_url(
    state_id, "https://spelunker.cave-explorer.org/"
)

# %%

# sample_object_predictions = object_predictions.sample(20)

colors = sns.color_palette("pastel", n_colors=5).as_hex()
for i, (label, ids) in enumerate(object_predictions.groupby(object_predictions)):
    # ids = ids.sample(max(1, len(ids)))
    sub_df = ids.to_frame().reset_index()
    sub_df["color"] = colors[i]
    new_seg_layer = statebuilder.SegmentationLayerConfig(
        source=client.info.segmentation_source(),
        name=label,
        selected_ids_column=grouping,
        color_column="color",
        alpha_3d=0.3,
    )
    sbs.append(statebuilder.StateBuilder(layers=[new_seg_layer]))
    dfs.append(sub_df)

sb = statebuilder.ChainedStateBuilder(sbs)

sb.render_state(dfs, return_as="html")
