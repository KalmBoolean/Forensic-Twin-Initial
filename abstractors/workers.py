from PySide6.QtCore import Signal, Slot, QRunnable, QObject
from insta import InstaSplatBoi
from gaussplat import GaussianSplat

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
            self.signals.statusChanged.emit("Running InstantSplat Initialiser...")

            output = self.imagePath.parent / "insta_output"

            inst = InstaSplatBoi(self.imagePath, output)
            inst.ClearData()
            inst.Build()

            self.signals.statusChanged.emit("Initializing Gaussians...")

            gaussianSplat = GaussianSplat(
                instInput=inst,
                image_path=self.imagePath,
                output_path=output / "gsplat",
                device="cuda",
                sh_degree=2,
            )

            self.signals.statusChanged.emit("Training Gaussians...")

            gaussianSplat.Train(iterations=1000)

            self.signals.finished.emit(gaussianSplat)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("\n========== BUILD ERROR ==========")
            print(type(e).__name__)
            print(e)
            print(tb)
            print("=================================\n")
            self.signals.failed.emit(str(e), tb)