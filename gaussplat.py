from pathlib import Path
from loader import load_gaussian_ply
import numpy as np
import torch
import torch.nn.functional as F

from gsplat import rasterization
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


    def Render(self, camera,custom_w2c=None):

        means = self.splats["means"]

        quats = F.normalize(
            self.splats["quats"],
            dim=-1,
        )

        scales = torch.exp(
            self.splats["scales"]
        )

        opacities = torch.sigmoid(
            self.splats["opacities"]
        )

        colors = torch.cat(
            [
                self.splats["sh0"],
                self.splats["shN"],
            ],
            dim=1,
        )
        if custom_w2c is None:
            viewmat = camera.world_view_transform.T
        else:
            viewmat = custom_w2c.unsqueeze(0)

        Ks = camera.K.unsqueeze(0)

        render, alpha, info = rasterization(

            means=means,

            quats=quats,

            scales=scales,

            opacities=opacities,

            colors=colors,

            viewmats=viewmat,
            Ks=Ks,

            width=camera.image_width,
            height=camera.image_height,

            sh_degree=self.shDegree,

        )

        return render[0], info

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