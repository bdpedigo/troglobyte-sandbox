# %%


from pathlib import Path

import caveclient as cc
import numpy as np
import pandas as pd

from troglobyte.features import CAVEWrangler

client = cc.CAVEclient("minnie65_phase3_v1")

out_path = Path("./troglobyte-sandbox/models/")

model_name = "local_compartment_classifier_bd_boxes"

data_path = Path("./troglobyte-sandbox/data/bounding_box_labels")

files = list(data_path.glob("*.csv"))

# %%
voxel_resolution = np.array([4, 4, 40])


def simple_labeler(label):
    if "axon" in label:
        return "axon"
    elif "dendrite" in label:
        return "dendrite"
    elif "glia" in label:
        return "glia"
    elif "soma" in label:
        return "soma"
    else:
        return np.nan


def axon_labeler(label):
    if label == "uncertain":
        return np.nan
    elif "axon" in label:
        return True
    else:
        return False


dfs = []
for file in files:
    file_label_df = pd.read_csv(file)

    file_label_df.set_index(["bbox_id", "root_id"], inplace=True, drop=False)
    file_label_df["ctr_pt_4x4x40"] = file_label_df["ctr_pt_4x4x40"].apply(
        lambda x: np.array(eval(x.replace("  ", ",").replace(" ", ",")), dtype=int)
    )

    file_label_df["x_nm"] = (
        file_label_df["ctr_pt_4x4x40"].apply(lambda x: x[0]) * voxel_resolution[0]
    )
    file_label_df["y_nm"] = (
        file_label_df["ctr_pt_4x4x40"].apply(lambda x: x[1]) * voxel_resolution[1]
    )
    file_label_df["z_nm"] = (
        file_label_df["ctr_pt_4x4x40"].apply(lambda x: x[2]) * voxel_resolution[2]
    )

    file_label_df["axon_label"] = file_label_df["label"].apply(axon_labeler)

    file_label_df["simple_label"] = file_label_df["label"].apply(simple_labeler)

    dfs.append(file_label_df)

# %%
label_df = pd.concat(dfs)
label_df.to_csv(out_path / model_name / "labels.csv")

# %%


points = label_df[["x_nm", "y_nm", "z_nm"]].values
neighborhood_hops = 5

# set up object
wrangler = CAVEWrangler(client, verbose=10, n_jobs=-1)

# list the objects we are interested in
wrangler.set_objects(label_df.index.get_level_values("root_id"))

# get a 20um bounding box around the points which were classified
wrangler.set_query_boxes_from_points(points, box_width=20_000)

# query the level2 ids for the objects in those bounding boxes
wrangler.query_level2_ids()

# query the level2 shape features for the objects in those bounding boxes
wrangler.query_level2_shape_features()

# query the level2 synapse features for the objects in those bounding boxes
# this uses the object IDs which were input for the synapse query, which may get out
# of date
wrangler.query_level2_synapse_features(method="update", chunk_size=250)

# # aggregate these features by k-hop neighborhoods in the level2 graph
# wrangler.aggregate_features_by_neighborhood(
#     aggregations=["mean", "std"],
#     neighborhood_hops=neighborhood_hops,
#     drop_self_in_neighborhood=True,
# )

wrangler.query_level2_edges()

# %%
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(wrangler.features_.fillna(0.0))
new_features = pd.DataFrame(
    X, index=wrangler.features_.index, columns=wrangler.features_.columns
)

# %%
wrangler.query_level2_networks(validate=False)
wrangler.object_level2_networks_

# %%
import torch
from tqdm.auto import tqdm

label_df["simple_label"] = label_df["simple_label"].astype("category")
labels = label_df.droplevel("bbox_id")["simple_label"]
labels = labels.dropna()

nfs = wrangler.object_level2_networks_.loc[labels.index]

# %%

from networkframe import NetworkFrame

one_hot_labels = pd.get_dummies(labels).astype(float)

graph_data = []
for root, nf in tqdm(nfs.items(), total=len(nfs)):
    nf = NetworkFrame(nf.nodes.copy(), nf.edges.copy())
    nf.nodes = new_features.loc[root].loc[nf.nodes.index]
    nf.nodes.drop(
        columns=[col for col in nf.nodes.columns if "rep_coord" in col], inplace=True
    )
    data = nf.to_torch_geometric()
    data.y = torch.tensor(
        one_hot_labels.loc[root].values.reshape(1, -1), dtype=torch.float16
    )
    graph_data.append(data)


# %%
import caveclient as cc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm.auto import tqdm

train_graphs, test_graphs = train_test_split(graph_data, test_size=0.2)

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)


# create a graph neural network model using PyTorch Geometric


N_NODE_FEATURES = len(graph_data[0].x[0])
N_CLASSES = 4


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.mlp = Linear(hidden_channels, hidden_channels)
        self.conv1 = GCNConv(N_NODE_FEATURES, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.mlp2 = Linear(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, N_CLASSES)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.mlp(x)
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.mlp(x)
        x = self.mlp2(x)
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.mlp(x)
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.mlp(x)
        x = self.conv4(x, edge_index)

        x = self.mlp2(x).relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=64)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        true = data.y.argmax(dim=1)
        correct += int((pred == true).sum())  # Check against ground-truth labels.

    acc = correct / len(loader.dataset)  # Derive ratio of correct predictions.
    loss = criterion(out.softmax(dim=1), data.y).item()
    return acc, loss


print(test(train_loader))

# %%
rows = []

import pprint

for epoch in tqdm(range(100)):
    train()
    train_acc, train_loss = test(train_loader)
    test_acc, test_loss = test(test_loader)
    rows.append(
        {
            "epoch": epoch,
            "acc": train_acc,
            "loss": train_loss,
            "evaluation": "train",
        }
    )
    rows.append(
        {
            "epoch": epoch,
            "acc": test_acc,
            "loss": test_loss,
            "evaluation": "test",
        }
    )
    if epoch % 10 == 0:
        pprint.pprint(rows[-2])
        pprint.pprint(rows[-1])
        print()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

progress_df = pd.DataFrame(rows)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
sns.lineplot(data=progress_df, x="epoch", y="acc", hue="evaluation", ax=axs[0])
sns.lineplot(data=progress_df, x="epoch", y="loss", hue="evaluation", ax=axs[1])

# %%
