# %%

import cloudvolume
from caveclient import CAVEclient

client = CAVEclient("minnie65_phase3_v1", version=661)


def get_root_id_from_point(point, voxel_resolution, client):
    cv = cloudvolume.CloudVolume(
        client.info.segmentation_source(),
        use_https=True,
        bounded=False,
        fill_missing=True,
        progress=False,
        secrets={"token": client.auth.token},
    )
    supervoxel = int(
        cv.download_point(
            point, size=1, coord_resolution=voxel_resolution, agglomerate=False
        )
        .squeeze()
        .item()
    )

    root_id = client.chunkedgraph.get_root_id(supervoxel)

    return root_id


get_root_id_from_point([284493.03125, 194604.21875, 19898.5015625], [4, 4, 40], client)

# %%

import numpy as np
import pandas as pd


def get_root_ids_from_points(points, voxel_resolution, client):
    cv = cloudvolume.CloudVolume(
        client.info.segmentation_source(),
        use_https=True,
        bounded=False,
        fill_missing=True,
        progress=False,
        secrets={"token": client.auth.token},
    )
    roots_by_point = cv.scattered_points(
        points,
        coord_resolution=voxel_resolution,
        agglomerate=True,
        timestamp=client.timestamp,
    )
    roots_by_point = pd.Series(roots_by_point).to_frame("root_id")
    roots_by_point.index.names = ["x", "y", "z"]
    roots_by_point.reset_index(inplace=True)
    seg_res = client.chunkedgraph.segmentation_info["scales"][0]["resolution"]
    factor = seg_res / np.array(voxel_resolution)
    roots_by_point[["x", "y", "z"]] *= factor

    return roots_by_point


points = np.array(
    [
        [284722.4375, 194311.671875, 19845.909375],
        [285178.75, 194183.90625, 19803.4890625],
    ]
)
get_root_ids_from_points(points, [4, 4, 40], client)

# %%

cv = cloudvolume.CloudVolume(
    client.info.segmentation_source(),
    use_https=True,
    bounded=False,
    fill_missing=True,
    progress=False,
    secrets={"token": client.auth.token},
)
