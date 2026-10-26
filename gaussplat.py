from pathlib import Path
from loader import load_gaussian_ply
import numpy as np
import torch
import torch.nn.functional as F

import math
from gaussian_splatting.diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from instantsplat.trainer import Trainer
from insta import InstaSplatBoi

class GaussianSplat:

    def __init__(
        self,
        instInput,
        image_path,
        output_path,
        device="cuda",
        sh_degree=2,
    ):
        self.lambda_dssim = 0.2
        self.instIn = instInput

        self.imgPath = Path(image_path)
        self.outPath = Path(output_path)

        self.device = torch.device(device)
        self.shDegree = sh_degree

        self.outPath.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.splats = None

    @classmethod
    def FromPLY(cls, path, device="cuda"):
        #alternate constructor, damn
        self = cls.__new__(cls)

        self.lambda_dssim = 0.2
        self.instIn = None
        self.imgPath = None
        self.outPath = Path(path).parent

        self.device = torch.device(device)

        splats, shDegree = load_gaussian_ply(path, device=device)

        self.splats = splats
        self.shDegree = shDegree

        return self
    def Initialize(self):
        self.splats = torch.nn.ParameterDict({

        "means": torch.nn.Parameter(
            self.instIn.gaussians._xyz.detach().clone()
        ),
        "quats": torch.nn.Parameter(
            self.instIn.gaussians._rotation.detach().clone()
        ),

        "scales": torch.nn.Parameter(
            self.instIn.gaussians._scaling.detach().clone()
        ),

        "opacities": torch.nn.Parameter(
            self.instIn.gaussians._opacity.detach().clone().squeeze(-1)
        ),

        "sh0": torch.nn.Parameter(
            self.instIn.gaussians._features_dc.detach().clone()
        ),

        "shN": torch.nn.Parameter(
            self.instIn.gaussians._features_rest.detach().clone()
        ),
    })

        self.shDegree = self.instIn.gaussians.max_sh_degree


    def Render(self, camera, custom_w2c=None):
        means = self.splats["means"]
        quats = F.normalize(self.splats["quats"], dim=-1)
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"]).unsqueeze(-1)

        shs = torch.cat(
            [self.splats["sh0"], self.splats["shN"]],
            dim=1
        ).contiguous()

        device = means.device
        dtype = means.dtype

        width = int(camera.image_width)
        height = int(camera.image_height)

        if custom_w2c is None:
            w2c = camera.view_matrix().to(device=device, dtype=dtype)
        else:
            w2c = custom_w2c.to(device=device, dtype=dtype)

        viewmat = camera.view_matrix_with_roll().to(device=device, dtype=dtype).T.contiguous()

        K = camera.K.to(device=device, dtype=dtype)

        tanfovx = width / (2.0 * K[0, 0].item())
        tanfovy = height / (2.0 * K[1, 1].item())

        znear = 0.01
        zfar = 100.0

        P = torch.zeros((4, 4), device=device, dtype=dtype)
        P[0, 0] = 1.0 / tanfovx
        P[1, 1] = 1.0 / tanfovy
        P[2, 2] = zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        P[3, 2] = 1.0

        projmat = P.T.contiguous()
        full_proj = viewmat @ projmat

        # Camera position in world coordinates
        campos = torch.linalg.inv(viewmat)[3, :3].contiguous()

        # Graphdeco rasterizer settings
        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros(3, device=device, dtype=dtype),
            scale_modifier=1.0,
            viewmatrix=viewmat,
            projmatrix=full_proj,
            sh_degree=int(self.shDegree),
            campos=campos,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

        rasterizer = GaussianRasterizer(
            raster_settings=raster_settings
        )

        # Screen-space means; required by Graphdeco's API
        means2D = torch.zeros_like(
            means,
            requires_grad=True
        )

        rendered_image, radii, depth_image = rasterizer(
            means3D=means,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=quats,
            cov3D_precomp=None,
        )

        # Convert [C, H, W] to [H, W, C]
        rendered_image = rendered_image.permute(1, 2, 0).contiguous()

        info = {
            "radii": radii,
            "depth": depth_image,
            "means2D": means2D,
            "visibility": radii > 0,
        }

        return rendered_image, info

    def Train(
        self,
        iterations=7000,
        callback=None,
    ):

        trainer = Trainer(
            self.instIn.gaussians,
            dataset=self.instIn.dataset,
        )

        for i in range(iterations):

            camera = self.instIn.dataset[i % len(self.instIn.dataset)]
            loss, out = trainer.step(camera)
            print(f"Step {i:5d} | Loss: {loss.item():.6f}")
        self.Initialize()


    def Save(self, filename="gaussians.pt"):

        path = self.outPath / filename

        state = {
            key: value.detach().cpu()
            for key, value in self.splats.items()
        }

        torch.save(
            state,
            path,
        )

        print(
            f"Saved Gaussian model to {path}"
        )

        return path