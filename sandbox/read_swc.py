# %%
import numpy as np
import pandas as pd
import pyvista as pv

pv.set_jupyter_backend('client')

def read_swc(file_path):
    # Read
    df = pd.read_csv(file_path, sep=" ", header=None, comment="#")
    df.columns = ["n", "type", "x", "y", "z", "r", "parent"]

    # Create points
    points = df[["x", "y", "z"]].values

    # Create lines in pyvista format
    lines = []
    for i, row in df.iterrows():
        if row["parent"] != -1:
            lines.append([row['n'], row["parent"]])
    lines = np.array(lines, dtype=int)
    indicator = np.full((lines.shape[0], 1), 2, dtype=int)
    pv_lines = np.hstack((indicator, lines))

    # Create pyvista object
    poly = pv.PolyData(points, lines=pv_lines)

    return poly


swc_path = "troglobyte-sandbox/sandbox/example.swc"

skeleton_poly = read_swc(swc_path)

plotter = pv.Plotter()

plotter.add_mesh(skeleton_poly, color='black', line_width=10)

plotter.show()

# %%
