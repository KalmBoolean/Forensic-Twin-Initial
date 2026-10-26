import sys
from pathlib import Path
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QFileDialog,
    QSplitter,
    QProgressBar,
    QMessageBox,
)

from viewport import Viewport
from insta import InstaSplatBoi
from gaussplat import GaussianSplat

class BuildWorker(QObject):
    statusChanged = Signal(str)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath

    def run(self):
        try:
            self.statusChanged.emit("Running InstantSplat Initialiser...")

            output = self.imagePath.parent / "insta_output"

            inst = InstaSplatBoi(self.imagePath, output)
            inst.ClearData()
            inst.Build()
            self.statusChanged.emit("Initializing Gaussians...")

            gaussianSplat = GaussianSplat(
                instInput=inst,
                image_path=self.imagePath,
                output_path=output / "gsplat",
                device="cuda",
                sh_degree=2,
            )

            self.statusChanged.emit("Training Gaussians...")

            gaussianSplat.Train(iterations=1100)

            self.finished.emit(gaussianSplat)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("\n========== BUILD ERROR ==========")
            print(type(e).__name__)
            print(e)
            print(tb)
            print("=================================\n")
            self.failed.emit(str(e), tb)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Forensic Digital Twin")
        self.resize(1400, 850)

        self.imagePath = None
        self.reconstruction = None
        self.gaussianSplat = None

        self.thread = None
        self.worker = None

        self.buildUi()

    def buildUi(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        panel = QWidget()
        panelLayout = QVBoxLayout(panel)

        title = QLabel("Scene Images")
        title.setStyleSheet(
            "font-size: 18px; font-weight: bold;"
        )

        self.folderLabel = QLabel(
            "No image folder selected"
        )
        self.folderLabel.setWordWrap(True)

        self.selectButton = QPushButton(
            "Select Image Folder"
        )
        self.selectButton.clicked.connect(
            self.selectImages
        )

        self.imageList = QListWidget()

        self.buildButton = QPushButton(
            "Build Scene"
        )
        self.buildButton.clicked.connect(
            self.buildScene
        )

        self.loadPlyButton = QPushButton(
            "Load .ply Gaussian File"
        )
        self.loadPlyButton.clicked.connect(
            self.loadPly
        )

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()

        self.statusLabel = QLabel("Ready")

        panelLayout.addWidget(title)
        panelLayout.addWidget(self.folderLabel)
        panelLayout.addWidget(self.selectButton)
        panelLayout.addWidget(self.imageList)
        panelLayout.addWidget(self.buildButton)
        panelLayout.addWidget(self.loadPlyButton)

        self.viewport = Viewport()

        splitter.addWidget(panel)
        splitter.addWidget(self.viewport)
        splitter.setSizes([300, 1100])

        layout.addWidget(splitter)
        layout.addWidget(self.statusLabel)
        layout.addWidget(self.progress)

    def selectImages(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder"
        )

        if not folder:
            return

        self.imagePath = Path(folder)
        self.folderLabel.setText(str(self.imagePath))
        self.imageList.clear()

        extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }

        images = sorted(
            p for p in self.imagePath.iterdir()
            if p.suffix.lower() in extensions
        )

        for image in images:
            self.imageList.addItem(image.name)

        self.statusLabel.setText(
            f"{len(images)} images loaded"
        )

    def buildScene(self):
        if self.imagePath is None:
            QMessageBox.warning(
                self,
                "No Images",
                "Select an image folder first."
            )
            return

        self.buildButton.setEnabled(False)
        self.selectButton.setEnabled(False)
        self.loadPlyButton.setEnabled(False)
        self.progress.show()
        self.statusLabel.setText("Starting build...")

        self.thread = QThread()
        self.worker = BuildWorker(self.imagePath)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.statusChanged.connect(self.statusLabel.setText)
        self.worker.finished.connect(self.onBuildFinished)
        self.worker.failed.connect(self.onBuildFailed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanupThread)

        self.thread.start()

    def onBuildFinished(self, gaussianSplat):
        self.gaussianSplat = gaussianSplat
        self.viewport.setSplat(self.gaussianSplat)
        self.statusLabel.setText("Gaussian scene ready")

    def onBuildFailed(self, message, tb):
        self.statusLabel.setText("Build failed")
        QMessageBox.critical(
            self,
            "Build Failed",
            f"{message}\n\nSee console for full traceback."
        )

    def cleanupThread(self):
        self.buildButton.setEnabled(True)
        self.selectButton.setEnabled(True)
        self.loadPlyButton.setEnabled(True)
        self.progress.hide()

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None

    def loadPly(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Gaussian .ply File",
            "",
            "PLY Files (*.ply)"
        )

        if not path:
            return

        self.statusLabel.setText(f"Loading {Path(path).name}...")
        QApplication.processEvents()

        try:
            self.gaussianSplat = GaussianSplat.FromPLY(
                path,
                device="cuda"
            )

            self.viewport.setSplat(self.gaussianSplat)

            self.statusLabel.setText(
                f"Loaded {Path(path).name}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()

            QMessageBox.critical(
                self,
                "Load Failed",
                f"Could not load .ply file:\n{e}"
            )
            self.statusLabel.setText("Load failed")


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()