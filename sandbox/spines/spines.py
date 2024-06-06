# %%
import numpy as np
import pandas as pd
import pyvista as pv
from caveclient import CAVEclient

from troglobyte.features import CAVEWrangler

client = CAVEclient("minnie65_phase3_v1")
cv = client.info.segmentation_cloudvolume()

target_labels = client.materialize.query_table("vortex_compartment_targets")

# %%
target_labels["point"] = target_labels["post_pt_position"].apply(
    lambda x: list(np.array(x) * np.array([4, 4, 40]))
)
sample_target_labels = target_labels.sample(50)

# %%
level2_ids = client.chunkedgraph.get_roots(
    sample_target_labels["post_pt_supervoxel_id"].values, stop_layer=2
)
level2_ids = pd.Index(level2_ids)

# %%
box_width = 5000
wrangler = CAVEWrangler(client)
wrangler.set_objects(sample_target_labels["post_pt_root_id"])
wrangler.set_query_boxes_from_points(
    sample_target_labels.set_index("post_pt_root_id")["point"],
    box_width=box_width,
)
wrangler.query_level2_ids()

# %%

# points = np.array(sample_target_labels["post_pt_position"].values.tolist()).astype(
#     float
# )
# points *= np.array([4, 4, 40])
# # %%
# level2_ids = wrangler.level2_ids_
# meshes_by_level2_id = {}
# for level2_id in level2_ids:
#     try:
#         mesh = cv.mesh.get(level2_id)
#         meshes_by_level2_id[level2_id] = mesh
#     except Exception as e:
#         print(e)

# %%
from cloudvolume import Bbox

meshes = {}
for root_id in sample_target_labels["post_pt_root_id"]:
    bounds = wrangler.find_boxes(root_id)[0].astype(float)
    bounds /= np.array([8, 8, 40])
    bounds = bounds.astype(int)
    bbox = Bbox(bounds[0], bounds[1])
    try:
        mesh = cv.mesh.get(root_id, bounding_box=bbox)
        meshes[root_id] = mesh
    except Exception as e:
        print(e)


# %%

points = sample_target_labels["point"].values

pv.set_jupyter_backend("trame")
shape = (5, 10)

plotter = pv.Plotter(shape=shape)

mesh_polys = {}
for i, (root_id, mesh) in enumerate(meshes.items()):
    mesh = mesh[root_id]
    vertices = mesh.vertices.copy().astype(float)
    shift = vertices.mean(axis=0)
    # vertices -= shift
    point = points[i].copy()
    # point -= shift
    faces = mesh.faces

    bounds = wrangler.find_boxes(root_id)[0].astype(float)
    # bounds -= shift

    box_bounds = [
        bounds[0, 0],
        bounds[1, 0],
        bounds[0, 1],
        bounds[1, 1],
        bounds[0, 2],
        bounds[1, 2],
    ]
    box_poly = pv.Box(
        bounds=box_bounds,
    )

    padded_faces = np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1)

    mesh_poly = pv.PolyData(vertices, faces=padded_faces)
    # mesh_poly = mesh_poly.clip_box(box_bounds, invert=False)
    mesh_poly = mesh_poly.clip_surface(box_poly)
    # mesh_poly = mesh_poly.connectivity(extraction_mode="closest", closest_point=point)

    mesh_poly = mesh_poly.decimate(0.9)
    mesh_polys[root_id] = mesh_poly

    point_poly = pv.PolyData(point)

    indices = np.unravel_index(i, shape)
    plotter.subplot(*indices)
    plotter.add_mesh(mesh_poly, color="grey")
    plotter.add_mesh(box_poly, style="wireframe", color="black")
    plotter.add_mesh(point_poly, color="red", point_size=10)
    classification = sample_target_labels.iloc[i]["tag"]
    # plotter.add_text(f"{classification}", font_size=26)
    actor = plotter.add_title(classification, font="courier", color="k", font_size=10)
    # break

# plotter.link_views()
plotter.show()

# %%

from sklearn.metrics import pairwise_distances

from networkframe import NetworkFrame

plotter = pv.Plotter(shape=shape)
nfs_by_root_id = {}

for i, (root_id, mesh) in enumerate(meshes.items()):
    mesh_poly = mesh_polys[root_id]
    point = points[i]
    point = np.array(point).astype(float)
    nodes = np.array(mesh_poly.points)
    faces = mesh_poly.faces.reshape(-1, 4)[:, 1:]
    edges = np.concatenate([faces[:, :2], faces[:, 1:], faces[:, ::2]])

    nodes = pd.DataFrame(nodes, columns=["x", "y", "z"])
    edges = pd.DataFrame(edges, columns=["source", "target"])

    nf = NetworkFrame(nodes, edges)
    nf.apply_node_features(columns=["x", "y", "z"], inplace=True)
    nf.edges["length"] = (
        np.linalg.norm(
            nf.edges[["source_x", "source_y", "source_z"]].values
            - nf.edges[["target_x", "target_y", "target_z"]].values,
            axis=1,
        )
        ** 2
    )

    dists_to_point = pairwise_distances(
        nf.nodes[["x", "y", "z"]].values, point.reshape(1, -1), metric="sqeuclidean"
    )[:, 0]
    nf.nodes["dist_to_point"] = dists_to_point

    nfs_by_root_id[root_id] = nf

    padded_lines = np.concatenate(
        [np.full((edges.shape[0], 1), 2), edges[["source", "target"]].values], axis=1
    )
    network_poly = pv.PolyData(nodes[["x", "y", "z"]].values, lines=padded_lines)
    network_poly["length"] = nf.edges["length"].values

    indices = np.unravel_index(i, shape)
    plotter.subplot(*indices)
    plotter.add_mesh(network_poly, scalars="length")
    point_poly = pv.PolyData(point)
    plotter.add_mesh(point_poly, color="red", point_size=10)
    classification = sample_target_labels.iloc[i]["tag"]
    actor = plotter.add_title(classification, font="courier", color="k", font_size=10)
plotter.show()

# %%
import torch

target_labels["category"] = sample_target_labels["tag"].astype("category")
target_label_dummies = pd.get_dummies(target_labels["category"]).astype(float)

sample_target_label_dummies = target_label_dummies.loc[sample_target_labels.index]
# target_labels['category_code'] = target_labels['category'].cat.codes

data_by_root_id = {}
for i, (root_id, nf) in enumerate(nfs_by_root_id.items()):
    nf: NetworkFrame
    data = nf.to_torch_geometric(
        directed=False, weight_col="length", node_columns=["dist_to_point"]
    )
    data.x = data.x / 1_000_000
    data.y = torch.tensor(
        sample_target_label_dummies.iloc[i].values.reshape(1, -1)
    ).float()
    data_by_root_id[root_id] = data

# %%

from torch_geometric.loader import DataLoader

data_list = list(data_by_root_id.values())
loader = DataLoader(data_list, batch_size=4, shuffle=True)

# %%
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.aggr import SumAggregation
from torch_scatter import scatter

# create a pytorch graph neural network which uses the edge weights


class EquivariantMPLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        act: nn.Module,
    ) -> None:
        super().__init__()
        self.act = act
        self.residual_proj = nn.Linear(in_channels, hidden_channels, bias=False)

        # Messages will consist of two (source and target) node embeddings and a scalar distance
        message_input_size = 2 * in_channels + 1

        # equation (3) "phi_l" NN
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input_size, hidden_channels),
            act,
        )
        # equation (4) "psi_l" NN
        self.node_update_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            act,
        )

    def node_message_function(
        self,
        source_node_embed: Tensor,  # h_i
        target_node_embed: Tensor,  # h_j
        node_dist: Tensor,  # d_ij
    ) -> Tensor:
        # implements equation (3)
        message_repr = torch.cat(
            (source_node_embed, target_node_embed, node_dist), dim=-1
        )
        print(message_repr.shape)
        return self.message_mlp(message_repr)

    # def compute_distances(self, node_pos: Tensor, edge_index: LongTensor) -> Tensor:
    #     row, col = edge_index
    #     xi, xj = node_pos[row], node_pos[col]
    #     # relative squared distance
    #     # implements equation (2) ||X_i - X_j||^2
    #     rsdist = (xi - xj).pow(2).sum(1, keepdim=True)
    #     return rsdist

    def forward(
        self,
        node_embed: Tensor,
        edge_dist: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        row, col = edge_index
        # dist = self.compute_distances(node_pos, edge_index)

        # compute messages "m_ij" from  equation (3)
        print(node_embed[row].shape, node_embed[col].shape, edge_dist.shape)
        node_messages = self.node_message_function(
            node_embed[row], node_embed[col], edge_dist
        )

        # message sum aggregation in equation (4)
        aggr_node_messages = scatter(node_messages, col, dim=0, reduce="sum")

        # compute new node embeddings "h_i^{l+1}"
        # (implements rest of equation (4))
        new_node_embed = self.residual_proj(node_embed) + self.node_update_mlp(
            torch.cat((node_embed, aggr_node_messages), dim=-1)
        )

        return new_node_embed


class EquivariantGNN(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        final_embedding_size: Optional[int] = None,
        target_size: int = 1,
        num_mp_layers: int = 2,
    ) -> None:
        super().__init__()
        if final_embedding_size is None:
            final_embedding_size = hidden_channels

        # non-linear activation func.
        # usually configurable, here we just use Relu for simplicity
        self.act = nn.ReLU()

        # equation (1) "psi_0"
        self.f_initial_embed = nn.Embedding(100, hidden_channels)

        # create stack of message passing layers
        self.message_passing_layers = nn.ModuleList()
        channels = [hidden_channels] * (num_mp_layers) + [final_embedding_size]
        for d_in, d_out in zip(channels[:-1], channels[1:]):
            layer = EquivariantMPLayer(d_in, d_out, self.act)
            self.message_passing_layers.append(layer)

        # modules required for readout of a graph-level
        # representation and graph-level property prediction
        self.aggregation = SumAggregation()
        self.f_predict = nn.Sequential(
            nn.Linear(final_embedding_size, final_embedding_size),
            self.act,
            nn.Linear(final_embedding_size, target_size),
        )

    def encode(self, data: Data) -> Tensor:
        # theory, equation (1)
        # node_embed = self.f_initial_embed(data.x)
        node_embed = data.x
        print(node_embed.shape)

        # message passing
        # theory, equation (3-4)
        for mp_layer in self.message_passing_layers:
            # NOTE here we use the complete edge index defined by the transform earlier on
            # to implement the sum over $j \neq i$ in equation (4)
            print(node_embed.shape, data.edge_attr.shape, data.edge_index.shape)
            node_embed = mp_layer(node_embed, data.edge_attr, data.edge_index)
        return node_embed

    def _predict(self, node_embed, batch_index) -> Tensor:
        aggr = self.aggregation(node_embed, batch_index)
        return self.f_predict(aggr)

    def forward(self, data: Data) -> Tensor:
        node_embed = self.encode(data)
        pred = self._predict(node_embed, data.batch)
        return pred


model = EquivariantGNN(hidden_channels=32, final_embedding_size=1, num_mp_layers=3, target_size=4)


from typing import Any, Callable, Dict, Tuple

from tqdm import tqdm


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    pbar: Optional[Any] = None,
    optim: Optional[torch.optim.Optimizer] = None,
):
    """Run a single epoch.

    Parameters
    ----------
    model : nn.Module
        the NN used for regression
    loader : DataLoader
        an iterable over data batches
    criterion : Callable[[Tensor, Tensor], Tensor]
        a criterion (loss) that is optimized
    pbar : Optional[Any], optional
        a tqdm progress bar, by default None
    optim : Optional[torch.optim.Optimizer], optional
        a optimizer that is optimizing the criterion, by default None
    """

    def step(
        data_batch: Data,
    ) -> Tuple[float, float]:
        """Perform a single train/val step on a data batch.

        Parameters
        ----------
        data_batch : Data

        Returns
        -------
        Tuple[float, float]
            Loss (mean squared error) and validation critierion (absolute error).
        """
        pred = model.forward(data_batch)
        target = data_batch.y
        loss = criterion(pred, target)
        if optim is not None:
            optim.zero_grad()
            loss.backward()
            optim.step()
        return loss.detach().item()

    if optim is not None:
        model.train()
        # This enables pytorch autodiff s.t. we can compute gradients
        model.requires_grad_(True)
    else:
        model.eval()
        # disable autodiff: when evaluating we do not need to track gradients
        model.requires_grad_(False)

    total_loss = 0
    total_mae = 0
    for data in loader:
        loss, mae = step(data)
        total_loss += loss * data.num_graphs
        total_mae += mae
        if pbar is not None:
            pbar.update(1)

    return total_loss / len(loader.dataset), total_mae / len(loader.dataset)


def train_model(
    data_module: Any,
    model: nn.Module,
    num_epochs: int = 30,
    lr: float = 3e-4,
    batch_size: int = 32,
    weight_decay: float = 1e-8,
    # best_model_path: Path = DATA.joinpath("trained_model.pth"),
) -> Dict[str, Any]:
    """Takes data and model as input and runs training, collecting additional validation metrics
    while doing so.

    Parameters
    ----------
    data_module : QM9DataModule
        a data module as defined earlier
    model : nn.Module
        a gnn model
    num_epochs : int, optional
        number of epochs to train for, by default 30
    lr : float, optional
        "learning rate": optimizer SGD step size, by default 3e-4
    batch_size : int, optional
        number of examples used for one training step, by default 32
    weight_decay : float, optional
        L2 regularization parameter, by default 1e-8
    best_model_path : Path, optional
        path where the model weights with lowest val. error should be stored
        , by default DATA.joinpath("trained_model.pth")

    Returns
    -------
    Dict[str, Any]
        a training result, ie statistics and info about the model
    """
    # create data loaders
    # train_loader = data_module.train_loader(batch_size=batch_size)
    train_loader = data_module
    # val_loader = data_module.val_loader(batch_size=batch_size)

    # setup optimizer and loss
    optim = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-8)
    loss_fn = nn.MSELoss()

    # keep track of the epoch with the best validation mae
    # st we can save the "best" model weights
    best_val_mae = float("inf")

    # Statistics that will be plotted later on
    # and model info
    result = {
        "model": model,
        # "path_to_best_model": best_model_path,
        "train_loss": np.full(num_epochs, float("nan")),
        "val_loss": np.full(num_epochs, float("nan")),
        "train_mae": np.full(num_epochs, float("nan")),
        "val_mae": np.full(num_epochs, float("nan")),
    }

    # Auxiliary functions for updating and reporting
    # Training progress statistics
    def update_statistics(i_epoch: int, **kwargs: float):
        for key, value in kwargs.items():
            result[key][i_epoch] = value

    def desc(i_epoch: int) -> str:
        return " | ".join(
            [f"Epoch {i_epoch + 1:3d} / {num_epochs}"]
            + [
                f"{key}: {value[i_epoch]:8.2f}"
                for key, value in result.items()
                if isinstance(value, np.ndarray)
            ]
        )

    # main training loop
    for i_epoch in range(0, num_epochs):
        progress_bar = tqdm(total=len(train_loader))
        try:
            # tqdm for reporting progress
            progress_bar.set_description(desc(i_epoch))

            # training epoch
            train_loss, train_mae = run_epoch(
                model, train_loader, loss_fn, progress_bar, optim
            )
            # validation epoch
            # val_loss, val_mae = run_epoch(model, val_loader, loss_fn, progress_bar)

            # update_statistics(
            #     i_epoch,
            #     train_loss=train_loss,
            #     # val_loss=val_loss,
            #     train_mae=train_mae,
            #     # val_mae=val_mae,
            # )

            progress_bar.set_description(desc(i_epoch))

            # if val_mae < best_val_mae:
            #     best_val_mae = val_mae
            # torch.save(model.state_dict(), best_model_path)
        finally:
            progress_bar.close()

    return result



train_model(loader, model, num_epochs=100, lr=1e-2, batch_size=4)

# %%
