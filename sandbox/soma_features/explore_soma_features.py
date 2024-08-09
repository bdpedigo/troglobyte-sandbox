#%%
import pandas as pd 
import pickle 
import numpy as np
from pathlib import Path

data_path = Path("troglobyte-sandbox/data/soma_features/microns_SomaData_AllCells_v661.pkl")

with open(data_path, "rb") as f:
    df = pickle.load(f)

