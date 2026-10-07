from plyfile import PlyData
import torch
import numpy as np


def load_gaussian_ply(path, device="cuda"):
    ply = PlyData.read(path)
    v = ply["vertex"]

    N = len(v)

    means = np.stack([
        v["x"],
        v["y"],
        v["z"]
    ], axis=1)

    quats = np.stack([
        v["rot_0"],
        v["rot_1"],
        v["rot_2"],
        v["rot_3"]
    ], axis=1)

    scales = np.stack([
        v["scale_0"],
        v["scale_1"],
        v["scale_2"]
    ], axis=1)

    opacities = np.asarray(v["opacity"])

    sh0 = np.stack([
        v["f_dc_0"],
        v["f_dc_1"],
        v["f_dc_2"]
    ], axis=1)

    sh0 = sh0[:, None, :]

    rest_names = [
        p.name
        for p in v.properties
        if p.name.startswith("f_rest_")
    ]

    rest_names.sort(
        key=lambda name: int(name.split("_")[-1])
    )

    if rest_names:
        shN = np.stack(
            [v[name] for name in rest_names],
            axis=1
        )

        shN = shN.reshape(N, 3, -1)
        shN = np.transpose(shN, (0, 2, 1))
    else:
        shN = np.empty(
            (N, 0, 3),
            dtype=np.float32
        )

    means = torch.tensor(
        means,
        dtype=torch.float32,
        device=device
    )

    quats = torch.tensor(
        quats,
        dtype=torch.float32,
        device=device
    )

    scales = torch.tensor(
        scales,
        dtype=torch.float32,
        device=device
    )

    opacities = torch.tensor(
        opacities,
        dtype=torch.float32,
        device=device
    ).unsqueeze(-1)

    sh0 = torch.tensor(
        sh0,
        dtype=torch.float32,
        device=device
    )

    shN = torch.tensor(
        shN,
        dtype=torch.float32,
        device=device
    )
    splats = torch.nn.ParameterDict({

        "means": torch.nn.Parameter(
            means
        ),

        "quats": torch.nn.Parameter(
            quats
        ),

        "scales": torch.nn.Parameter(
            scales
        ),

        "opacities": torch.nn.Parameter(
            opacities.squeeze(-1)
        ),

        "sh0": torch.nn.Parameter(
            sh0
        ),

        "shN": torch.nn.Parameter(
            shN
        ),
    })

    num_sh_coeffs = 1 + shN.shape[1]

    shDegree = int(np.sqrt(num_sh_coeffs) - 1)

    return splats, shDegree