# %%
from io import BytesIO

import pandas as pd
from cloudfiles import CloudFiles

cf = CloudFiles("gs://allen-minnie-phase3/vasculature_feature_pulls/segclr/2024-08-19")

file_names = list(cf.list())
print(file_names[:20])

# %%
def load_dataframe(path, **kwargs):
    bytes_out = cf.get(path)
    with BytesIO(bytes_out) as f:
        df = pd.read_csv(f, **kwargs)
    return df


for file_name in file_names:
    if "segclr_features" in file_name:
        segclr_features = load_dataframe(file_name, index_col=[0, 1])
        break
# %%
segclr_features

#%%
level2_features = load_dataframe(file_names[0], index_col=[0, 1])

level2_features.groupby('root_id')['size_nm3'].sum()