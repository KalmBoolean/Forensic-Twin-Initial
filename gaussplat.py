from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pycolmap
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset.colmap import colmap_init
from gaussian_splatting.trainer import BaseOpacityResetDensificationTrainer
from gaussian_splatting.dataset.colmap import ColmapCameraDataset
from gaussian_splatting.diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


class GaussianSplat:

    def __init__(
        self,
        reconstruction,
        image_path,
        output_path,
        device="cuda",
        sh_degree=2,
    ):
        self.lambda_dssim = 0.2
        self.reconstruction = reconstruction

        self.imgPath = Path(image_path)
        self.outPath = Path(output_path)

        self.device = torch.device(device)
        self.shDegree = sh_degree

        self.outPath.mkdir(parents=True, exist_ok=True)

        self.splats = None

    @classmethod
    def FromPLY(cls, path, device="cuda", sh_degree=2):

        self = cls.__new__(cls)

        self.lambda_dssim = 0.2
        self.reconstruction = None
        self.imgPath = None
        self.outPath = None

        self.device = torch.device(device)
        self.shDegree = sh_degree

        self.splats = GaussianModel(sh_degree).to(self.device)
        self.splats.load_ply(str(path))

        return self

    def Initialize(self):
        self.splats = GaussianModel(self.shDegree).to(self.device)
        scenePath = self.imgPath.parent
        sparsePath = scenePath / "sparse" / "0"
        self.undistortedPath = scenePath / "undistorted"
        sparseDir = self.undistortedPath / "sparse"
        
        
        if not self.undistortedPath.exists():
            pycolmap.undistort_images(
                output_path=str(self.undistortedPath),
                input_path=str(sparsePath),
                image_path=str(self.imgPath),
                output_type="COLMAP"
            )
        
        modelDir = sparseDir / "0"
        if not modelDir.exists():
            modelDir.mkdir(parents=True)
        
        for filePath in sparseDir.glob("*.bin"):
            filePath.rename(modelDir / filePath.name)
            
        colmap_init(
            self.splats,
            str(self.undistortedPath)
        )

    def Render(self, camera, custom_w2c=None):

        means = self.splats.get_xyz
        quats = self.splats.get_rotation
        scales = self.splats.get_scaling
        opacities = self.splats.get_opacity

        shs = self.splats.get_features.contiguous()

        device = means.device
        dtype = means.dtype

        width = int(camera.image_width)
        height = int(camera.image_height)

        if custom_w2c is None:
            w2c = camera.view_matrix().to(device=device, dtype=dtype)
        else:
            w2c = custom_w2c.to(device=device, dtype=dtype)

        viewmat = camera.view_matrix_with_roll().to(
            device=device,
            dtype=dtype
        ).T.contiguous()

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

        campos = torch.linalg.inv(viewmat)[3, :3].contiguous()

        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros(3, device=device, dtype=dtype),
            scale_modifier=1.0,
            viewmatrix=viewmat,
            projmatrix=full_proj,
            sh_degree=int(self.splats.active_sh_degree),
            campos=campos,
            prefiltered=False,
            debug=False,
            antialiasing=False
        )

        rasterizer = GaussianRasterizer(
            raster_settings=raster_settings
        )

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

        rendered_image = rendered_image.permute(
            1, 2, 0
        ).contiguous()

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
        

        dataset = ColmapCameraDataset(str(self.undistortedPath))

        if self.splats is None:
            self.Initialize()

        trainer = BaseOpacityResetDensificationTrainer(
            self.splats,
            dataset=dataset,
        )

        for i in range(iterations):

            camera = dataset[i % len(dataset)]

            loss, out = trainer.step(camera)

            if callback is not None:
                callback(i, loss)

            
            print(
                    f"Step {i:5d} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Gaussians: {len(self.splats.get_xyz)}"
            )

        return self.splats

    def Save(self, path):

        self.splats.save_ply(str(path))
