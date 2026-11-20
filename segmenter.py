
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


class SamSegmenter:
    def __init__(self, modelId="facebook/sam2.1-hiera-tiny", device="auto", maxSide=1024, pointsPerBatch=8, predIouThreshold=0.8, stabilityThreshold=0.88, minArea=100, maxMasks=0):
        self.modelId = modelId
        self.device = 0 if device == "auto" and torch.cuda.is_available() else -1 if device == "auto" or device == "cpu" else 0
        self.maxSide = maxSide
        self.pointsPerBatch = pointsPerBatch
        self.predIouThreshold = predIouThreshold
        self.stabilityThreshold = stabilityThreshold
        self.minArea = minArea
        self.maxMasks = maxMasks
        self.generator = None

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        if device not in ("auto", "cuda", "cpu"):
            raise ValueError("Invalid device.")

    def Load(self):
        if self.generator is None:
            from transformers import pipeline

            modelDtype = torch.float16 if self.device >= 0 else torch.float32

            self.generator = pipeline(
                "mask-generation",
                model=self.modelId,
                device=self.device,
                model_kwargs={"torch_dtype": modelDtype}
            )

    def SegmentImage(self, image):
        self.Load()

        if isinstance(image, (str, Path)):
            imagePath = Path(image)

            with Image.open(imagePath) as openedImage:
                originalImage = ImageOps.exif_transpose(openedImage).convert("RGB")

            source = str(imagePath)
        else:
            originalImage = ImageOps.exif_transpose(image).convert("RGB")
            source = None

        width, height = originalImage.size
        inferenceImage = originalImage

        if self.maxSide > 0 and max(width, height) > self.maxSide:
            scale = self.maxSide / max(width, height)
            newSize = (round(width * scale), round(height * scale))
            inferenceImage = originalImage.resize(newSize, Image.Resampling.BILINEAR)

        with torch.inference_mode():
            result = self.generator(
                inferenceImage,
                points_per_batch=self.pointsPerBatch,
                pred_iou_thresh=self.predIouThreshold,
                stability_score_thresh=self.stabilityThreshold
            )

        rawMasks = result.get("masks", [])
        rawScores = result.get("scores", [])

        if isinstance(rawScores, torch.Tensor):
            rawScores = rawScores.detach().cpu().numpy()

        masks = []
        scores = []

        for index, rawMask in enumerate(rawMasks):
            if isinstance(rawMask, torch.Tensor):
                rawMask = rawMask.detach().cpu().numpy()

            mask = np.asarray(rawMask).squeeze() > 0

            if mask.shape != (height, width):
                maskImage = Image.fromarray(mask.astype(np.uint8) * 255)
                maskImage = maskImage.resize(
                    (width, height),
                    Image.Resampling.NEAREST
                )
                mask = np.asarray(maskImage) > 0

            area = int(mask.sum())

            if area < self.minArea:
                continue
            masks.append(mask)
            scores.append(float(rawScores[index]) if index < len(rawScores) else 1.0)

        order = np.argsort(scores)[::-1]

        if self.maxMasks > 0:
            order = order[:self.maxMasks]

        if len(order):
            masks = np.stack([masks[index] for index in order])
        else:
            masks = np.zeros((0, height, width), dtype=bool)

        scores = np.asarray([scores[index] for index in order], dtype=np.float32)
        areas = masks.reshape(len(masks), -1).sum(axis=1).astype(np.int64)

        return {
            "masks": masks,
            "scores": scores,
            "areas": areas,
            "imageSize": (width, height),
            "source": source
        }
