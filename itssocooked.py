
import torch
import torch.nn.functional as F


class GaussianProjector:
    def __init__(self, splat):
        self.splat = splat
        self.device = splat.device

    def Project(self, camera):
        means = self.splat.splats["means"].detach()
        scales = torch.exp(self.splat.splats["scales"].detach())
        rotations = F.normalize(
            self.splat.splats["quats"].detach(),
            dim=-1
        )
        opacities = torch.sigmoid(
            self.splat.splats["opacities"].detach()
        ).reshape(-1)

        device = means.device
        dtype = means.dtype

        width = int(camera.image_width)
        height = int(camera.image_height)

        w2c = camera.view_matrix().to(device=device, dtype=dtype)
        K = camera.K.to(device=device, dtype=dtype)

        gaussianIdMap = torch.full(
            (height, width), -1, dtype=torch.int32, device=device
        )
        importanceMap = torch.zeros(
            (height, width), device=device, dtype=dtype
        )
        transmittanceMap = torch.ones(
            (height, width), device=device, dtype=dtype
        )

        def EmptyResult():
            return {
                "ID": gaussianIdMap,
                "Importance": importanceMap,
                "Transmittance": transmittanceMap
            }

        cameraPoints = means @ w2c[:3, :3].T + w2c[:3, 3]

        originalIds = torch.arange(len(means), device=device)

        valid = (
            torch.isfinite(means).all(dim=-1)
            & torch.isfinite(scales).all(dim=-1)
            & torch.isfinite(rotations).all(dim=-1)
            & torch.isfinite(opacities)
            & torch.isfinite(cameraPoints).all(dim=-1)
            & (scales > 0).all(dim=-1)
            & (cameraPoints[:, 2] > 0.01)
        )

        means = means[valid]
        scales = scales[valid]
        rotations = rotations[valid]
        opacities = opacities[valid]
        cameraPoints = cameraPoints[valid]
        originalIds = originalIds[valid]

        if len(means) == 0:
            return EmptyResult()

        x, y, z = cameraPoints.unbind(dim=-1)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        pixelX = fx * x / z + cx
        pixelY = fy * y / z + cy

        w, qx, qy, qz = rotations.unbind(dim=-1)

        rotation = torch.stack([
            1 - 2 * (qy.square() + qz.square()),
            2 * (qx * qy - qz * w),
            2 * (qx * qz + qy * w),

            2 * (qx * qy + qz * w),
            1 - 2 * (qx.square() + qz.square()),
            2 * (qy * qz - qx * w),

            2 * (qx * qz - qy * w),
            2 * (qy * qz + qx * w),
            1 - 2 * (qx.square() + qy.square())
        ], dim=-1).reshape(-1, 3, 3)

        covarianceWorld = (
            rotation
            @ torch.diag_embed(scales.square())
            @ rotation.transpose(1, 2)
        )

        cameraRotation = w2c[:3, :3]

        covarianceCamera = (
            cameraRotation.unsqueeze(0)
            @ covarianceWorld
            @ cameraRotation.T.unsqueeze(0)
        )

        jacobian = torch.zeros(
            (len(means), 2, 3),
            device=device,
            dtype=dtype
        )

        jacobian[:, 0, 0] = fx / z
        jacobian[:, 0, 2] = -fx * x / z.square()

        jacobian[:, 1, 1] = fy / z
        jacobian[:, 1, 2] = -fy * y / z.square()

        covariance2D = (
            jacobian
            @ covarianceCamera
            @ jacobian.transpose(1, 2)
        )

        covariance2D = 0.5 * (
            covariance2D + covariance2D.transpose(1, 2)
        )

        covariance2D[:, 0, 0] += 0.3
        covariance2D[:, 1, 1] += 0.3

        validCovariance = (
            torch.isfinite(covariance2D).all(dim=(1, 2))
            & torch.isfinite(pixelX)
            & torch.isfinite(pixelY)
        )

        covariance2D = covariance2D[validCovariance]
        pixelX = pixelX[validCovariance]
        pixelY = pixelY[validCovariance]
        z = z[validCovariance]
        opacities = opacities[validCovariance]
        originalIds = originalIds[validCovariance]

        if len(z) == 0:
            return EmptyResult()

        a = covariance2D[:, 0, 0]
        b = covariance2D[:, 0, 1]
        c = covariance2D[:, 1, 1]

        discriminant = torch.sqrt(
            ((a - c).square() + 4.0 * b.square()).clamp(min=0)
        )

        largestEigenvalue = 0.5 * (a + c + discriminant)

        radii = 3.0 * torch.sqrt(
            largestEigenvalue.clamp(min=1e-6)
        )

        determinant = a * c - b.square()

        valid = (
            torch.isfinite(radii)
            & torch.isfinite(determinant)
            & (determinant > 1e-12)
            & (pixelX + radii >= 0)
            & (pixelX - radii < width)
            & (pixelY + radii >= 0)
            & (pixelY - radii < height)
        )

        covariance2D = covariance2D[valid]
        pixelX = pixelX[valid]
        pixelY = pixelY[valid]
        z = z[valid]
        radii = radii[valid]
        opacities = opacities[valid]
        originalIds = originalIds[valid]

        if len(z) == 0:
            return EmptyResult()

        a = covariance2D[:, 0, 0]
        b = covariance2D[:, 0, 1]
        c = covariance2D[:, 1, 1]

        determinant = a * c - b.square()

        covarianceInverse = torch.zeros_like(covariance2D)
        covarianceInverse[:, 0, 0] = c / determinant
        covarianceInverse[:, 0, 1] = -b / determinant
        covarianceInverse[:, 1, 0] = -b / determinant
        covarianceInverse[:, 1, 1] = a / determinant

        sortedOrder = torch.argsort(z)

        covarianceInverse = covarianceInverse[sortedOrder]
        pixelX = pixelX[sortedOrder]
        pixelY = pixelY[sortedOrder]
        radii = radii[sortedOrder]
        opacities = opacities[sortedOrder]
        originalIds = originalIds[sortedOrder]

        for index in range(len(originalIds)):
            centerX = pixelX[index]
            centerY = pixelY[index]
            radius = radii[index]

            xMin = max(0, int(torch.floor(centerX - radius).item()))
            xMax = min(width, int(torch.ceil(centerX + radius).item()))

            yMin = max(0, int(torch.floor(centerY - radius).item()))
            yMax = min(height, int(torch.ceil(centerY + radius).item()))

            if xMin >= xMax or yMin >= yMax:
                continue

            gridY, gridX = torch.meshgrid(
                torch.arange(yMin, yMax, device=device, dtype=dtype),
                torch.arange(xMin, xMax, device=device, dtype=dtype),
                indexing="ij"
            )

            dx = gridX - centerX
            dy = gridY - centerY

            covarianceInv = covarianceInverse[index]

            power = -0.5 * (
                covarianceInv[0, 0] * dx.square()
                + 2 * covarianceInv[0, 1] * dx * dy
                + covarianceInv[1, 1] * dy.square()
            )

            alpha = (
                opacities[index] * torch.exp(power)
            ).clamp(min=0, max=0.99)

            alpha = torch.where(
                alpha >= 1.0 / 255.0,
                alpha,
                torch.zeros_like(alpha)
            )

            transmittance = transmittanceMap[
                yMin:yMax, xMin:xMax
            ]

            contribution = transmittance * alpha

            importance = importanceMap[
                yMin:yMax, xMin:xMax
            ]

            better = contribution > importance

            importanceMap[yMin:yMax, xMin:xMax] = torch.where(
                better,
                contribution,
                importance
            )

            currentIds = gaussianIdMap[
                yMin:yMax, xMin:xMax
            ]

            gaussianIdMap[yMin:yMax, xMin:xMax] = torch.where(
                better,
                originalIds[index].to(torch.int32),
                currentIds
            )

            transmittanceMap[yMin:yMax, xMin:xMax] = (
                transmittance * (1.0 - alpha)
            )

        return {
            "ID": gaussianIdMap,
            "Importance": importanceMap,
            "Transmittance": transmittanceMap
        }

    def LiftMask(self, mask, gaussianIdMap, importanceMap, minImportance=0.01):
        mask = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device=self.device
        )

        valid = (
            mask &
            (gaussianIdMap >= 0) &
            (importanceMap >= minImportance)
        )

        gaussianIds = gaussianIdMap[valid].long()
        weights = importanceMap[valid]

        uniqueIds, inverse = torch.unique(
            gaussianIds,
            return_inverse=True
        )

        gaussianWeights = torch.zeros(
            len(uniqueIds),
            device=self.device
        )

        gaussianWeights.scatter_add_(
            0,
            inverse,
            weights
        )

        return uniqueIds, gaussianWeights
    

