from PySide6.QtCore import Signal, QRunnable, QObject
from gaussplat import GaussianSplat
import pycolmap
from glostuff import GlobalMap
from pathlib import Path
from PySide6.QtCore import QRunnable, QObject, Signal
from gaussplat import GaussianSplat
import traceback

class WorkerSignals(QObject):
    statusChanged = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)


class BuildWorker(QRunnable):

    signals = WorkerSignals()

    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath

    def run(self):
        try:
            output = self.imagePath.parent

            self.signals.statusChanged.emit(
                "Extracting features and running GLOMAP..."
            )

            GlobalMap(
                outputDir=output,
                imageDir=self.imagePath
            )

            self.signals.statusChanged.emit(
                "Loading GLOMAP reconstruction..."
            )

            sparsePath = output / "sparse" / "0"

            reconstruction = pycolmap.Reconstruction(
                str(sparsePath)
            )
            gsOutput = output / "gsplat"

            self.signals.statusChanged.emit(
                "Initializing Gaussians..."
            )

            gaussianSplat = GaussianSplat(
                reconstruction=reconstruction,
                image_path=self.imagePath,
                output_path=gsOutput,
                device="cuda",
                sh_degree=2,
            )

            gaussianSplat.Initialize()

            self.signals.statusChanged.emit(
                "Training Gaussians..."
            )

            gaussianSplat.Train(
                iterations=7000
            )
            gaussianSplat.Save(output / "output" / "gsplat.ply")
            self.signals.finished.emit(gaussianSplat)

        except Exception as e:
            tb = traceback.format_exc()

            print("\n========== BUILD ERROR ==========")
            print(type(e).__name__)
            print(e)
            print(tb)
            print("=================================\n")

            self.signals.failed.emit(str(e), tb)

class FolderBuildWorker(QRunnable):

    def __init__(self, buildPath):
        super().__init__()

        self.buildPath = Path(buildPath)
        self.imagesPath = self.buildPath / "images"
        self.sparsePath = self.buildPath / "sparse" / "0"

        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.statusChanged.emit("Loading GLOMAP reconstruction")

            reconstruction = pycolmap.Reconstruction(
                str(self.sparsePath)
            )

            self.signals.statusChanged.emit("Initializing Gaussian Splatting")

            gaussianSplat = GaussianSplat(
                reconstruction=reconstruction,
                image_path=self.imagesPath,
                output_path=self.buildPath / "gsplat",
                device="cuda",
                sh_degree=2
            )

            gaussianSplat.Initialize()

            self.signals.statusChanged.emit("Training Gaussian Splatting")

            gaussianSplat.Train(iterations=20000)
            gaussianSplat.Save(self.buildPath/"output"/"gsplat.ply")
            self.signals.finished.emit(gaussianSplat)

        except Exception as e:
            tb = traceback.format_exc()
            
            print("\n========== BUILD ERROR ==========")
            print(type(e).__name__)
            print(e)
            print(tb)
            print("=================================\n")
            
            self.signals.failed.emit(str(e), tb)