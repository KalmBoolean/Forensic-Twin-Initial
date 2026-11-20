
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from contextlib import nullcontext
from transformers import AutoImageProcessor, AutoModel


class DINOFeatureExtractor:
    def __init__(self, modelId="facebook/dinov2-small", device="cuda", maxSide=518):
        self.modelId = modelId
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.maxSide = maxSide
        self.processor = None
        self.model = None

        if device not in ("cuda", "cpu"):
            raise ValueError("Invalid device.")

    def Load(self):
        if self.model is None:
            self.processor = AutoImageProcessor.from_pretrained(self.modelId)
            self.model = AutoModel.from_pretrained(self.modelId)
            self.model.to(self.device)
            self.model.eval()

    def ExtractFeatures(self, image, masks):
        self.Load()

        if isinstance(image, (str, Path)):
            with Image.open(image) as openedImage:
                image = ImageOps.exif_transpose(openedImage).convert("RGB")
        else:
            image = ImageOps.exif_transpose(image).convert("RGB")

        width, height = image.size
        patchSize = self.model.config.patch_size

        if self.maxSide > 0 and max(width, height) > self.maxSide:
            scale = self.maxSide / max(width, height)
        else:
            scale = 1.0

        newWidth = max(patchSize, round(width * scale / patchSize) * patchSize)
        newHeight = max(patchSize, round(height * scale / patchSize) * patchSize)

        image = image.resize((newWidth, newHeight), Image.Resampling.BICUBIC)

        imageArray = np.asarray(image).astype(np.float32) / 255.0
        imageTensor = torch.from_numpy(imageArray).permute(2, 0, 1)

        mean = torch.tensor(self.processor.image_mean).view(3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(3, 1, 1)

        imageTensor = (imageTensor - mean) / std
        imageTensor = imageTensor.unsqueeze(0).to(self.device)

        context = torch.autocast("cuda", dtype=torch.float16) if self.device == "cuda" else nullcontext()

        with torch.inference_mode(), context:
            output = self.model(
                pixel_values=imageTensor,
                interpolate_pos_encoding=True
            )

        patchFeatures = output.last_hidden_state[:, 1:, :]

        gridHeight = newHeight // patchSize
        gridWidth = newWidth // patchSize

        featureMap = patchFeatures.reshape(
            1, gridHeight, gridWidth, -1
        )[0]

        featureMap = F.normalize(featureMap.float(), dim=-1)

        maskTensor = torch.as_tensor(
            np.asarray(masks),
            dtype=torch.float32,
            device=self.device
        )

        if maskTensor.ndim != 3:
            raise ValueError("Masks must have shape (N, H, W).")

        if maskTensor.shape[1:] != (height, width):
            raise ValueError("Mask dimensions must match the original image.")

        maskTensor = F.interpolate(
            maskTensor.unsqueeze(1),
            size=(gridHeight, gridWidth),
            mode="area"
        )[:, 0]

        features = []
        allPatchFeatures = []
        allPatchWeights = []

        for mask in maskTensor:
            selected = mask > 0.05

            if not selected.any():
                features.append(torch.zeros(featureMap.shape[-1]))
                allPatchFeatures.append(torch.empty((0, featureMap.shape[-1])))
                allPatchWeights.append(torch.empty(0))
                continue

            selectedFeatures = featureMap[selected]
            selectedWeights = mask[selected]

            meanFeature = (
                selectedFeatures * selectedWeights.unsqueeze(-1)
            ).sum(dim=0) / selectedWeights.sum()

            meanFeature = F.normalize(meanFeature, dim=0)

            features.append(meanFeature.cpu())
            allPatchFeatures.append(selectedFeatures.cpu())
            allPatchWeights.append(selectedWeights.cpu())

        return {
            "features": torch.stack(features).numpy(),
            "patchFeatures": [x.numpy() for x in allPatchFeatures],
            "patchWeights": [x.numpy() for x in allPatchWeights],
            "gridSize": (gridWidth, gridHeight),
            "imageSize": (width, height)
        }
