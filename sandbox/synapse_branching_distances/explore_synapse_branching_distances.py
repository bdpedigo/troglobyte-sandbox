# %%

from caveclient import CAVEclient

# Initialize a client for the "minnie65_public" datastack.
client = CAVEclient(datastack_name="minnie65_public", version=1078)

proofreading_table = client.materialize.query_table("proofreading_status_and_strategy", filter_equal_dict={"status": "finalized"})

#%%
from pcg_skel import pcg_skeleton

)