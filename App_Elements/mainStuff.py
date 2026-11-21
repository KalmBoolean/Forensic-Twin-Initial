
import os
from pathlib import Path

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QSplitter,
    QProgressBar,
    QMessageBox,
    QInputDialog
)
import shutil
from .projectitems import *

from abstractors.workers import BuildWorker
from .leftpanel import LeftPanel
from .viewportarea import ViewportArea
from .rightpanel import RightPanel
class SceneBuilder(QWidget):

    imageExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    threadpool = QThreadPool()

    def __init__(self, workspacePath,parent=None):
        super().__init__(parent)
        self.workspace_path = workspacePath
        self.worker = None
        self.viewportArea = ViewportArea()

        self.leftPanel = LeftPanel(
            viewport=self.viewportArea,
            on_new_build=self.CreateBuild,
            threadpool=self.threadpool,
            update_status=self.BuildStartUIStuff,
            build_failed=self.onBuildFailed,
            build_finished=self.onBuildFinished
        )
        self.rightPanel = RightPanel()
        self.buildUi()

    def buildUi(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.leftPanel)
        splitter.addWidget(self.viewportArea)
        splitter.addWidget(self.rightPanel)
        splitter.setSizes([250, 600,250])

        layout.addWidget(splitter,1)

        self.statusLabel = QLabel("Ready")

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()

        layout.addWidget(self.statusLabel)
        layout.addWidget(self.progress)

    def selectImages(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder"
        )

        if not folder:
            return

        imagePath = Path(folder)
        imageList = []

        images = sorted(
            p for p in imagePath.iterdir()
            if p.suffix.lower() in self.imageExtensions
        )

        for image in images:
            imageList.append(image)

        return imageList,imagePath

    def buildScene(self,imagePath):
        if imagePath is None:
            QMessageBox.warning(
                self,
                "No Images",
                "Select an image folder first."
            )
            return

        self.BuildStartUIStuff("Starting Build...")

        self.worker = BuildWorker(imagePath)

        self.worker.signals.statusChanged.connect(
            self.BuildStartUIStuff
        )

        self.worker.signals.finished.connect(
            self.onBuildFinished
        )

        self.worker.signals.failed.connect(
            self.onBuildFailed
        )

        self.threadpool.start(self.worker)

    def BuildStartUIStuff(self,txt):
        self.leftPanel.new_build_button.setEnabled(False)
        self.progress.show()
        self.statusLabel.setText(txt)
        
    def onBuildFinished(self, gaussianSplat):
        self.gaussianSplat = gaussianSplat

        try:

            self.GetProjectItems(self.buildPath)

            self.viewportArea.show_splat(
                self.gaussianSplat, "Gaussian Splat"
            )

            self.statusLabel.setText("Gaussian scene ready")

        except Exception as e:
            QMessageBox.critical(self, "Build Failed", str(e))

        finally:
            self.leftPanel.new_build_button.setEnabled(True)
            self.progress.hide()

    def onBuildFailed(self, message, tb):
        self.statusLabel.setText("Build failed")

        QMessageBox.critical(
            self,
            "Build Failed",
            f"{message}\n\nSee console for full traceback."
        )

        self.leftPanel.new_build_button.setEnabled(True)
        self.progress.hide()


    def LoadWorkspace(self):
        self.leftPanel.clear_project()

        workspace_path = Path(self.workspace_path)

        for item_path in sorted(workspace_path.iterdir()):
            if item_path.is_dir():
                self.GetProjectItems(item_path)


    def GetProjectItems(self, build_path):

        build_item = FolderItem(
            build_path.name,
            build_path,
            self.leftPanel.root
        )

        images_path = build_path / "images"
        output_path = build_path / "output"

        images_folder = FolderItem(
            "Images", images_path, build_item
        )

        output_folder = FolderItem(
            "Output", output_path, build_item
        )

        if images_path.exists():
            for path in sorted(images_path.iterdir()):
                if path.suffix.lower() in self.imageExtensions:
                    ImageItem(path, images_folder)

        if output_path.exists():
            for path in sorted(output_path.glob("*.ply")):
                PlyItem(path, output_folder)

        build_item.setExpanded(True)
        images_folder.setExpanded(True)
        output_folder.setExpanded(True)

        return build_item


    def CreateBuild(self):
        name, ok = QInputDialog.getText(
            self, "New Build", "Enter build name:"
        )

        if not ok or not name.strip():
            return

        name = name.strip()

        if any(c in name for c in '<>:"/\\|?*'):
            QMessageBox.warning(self, "Invalid Name", "Invalid build name.")
            return

        build_path = Path(self.workspace_path) / name

        if build_path.exists():
            QMessageBox.warning(self, "Build Exists", "Build already exists.")
            return

        selected = self.selectImages()

        if not selected:
            return

        images, imagePath = selected

        if not images:
            QMessageBox.warning(self, "No Images", "No supported images found.")
            return

        try:
            self.buildPath = build_path
            self.outputPath = build_path / "output"

            images_path = build_path / "images"

            images_path.mkdir(parents=True)
            self.outputPath.mkdir()

            for image in images:
                shutil.copy2(image, images_path / image.name)

            self.buildScene(images_path)

        except Exception as e:
            if build_path.exists():
                shutil.rmtree(build_path)

            QMessageBox.critical(self, "Build Creation Failed", str(e))
    def SetWorkspacePath(self,path):
        self.workspace_path = path
        self.LoadWorkspace()
