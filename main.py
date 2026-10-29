import os
import sys
import traceback
from pathlib import Path

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

from PySide6.QtCore import Qt, QObject, QSize, QThread, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from viewport import Viewport
from insta import InstaSplatBoi
from gaussplat import GaussianSplat

from workspace import IMAGE_EXTS, Workspace, WorkspaceManager
from explorer import ProjectExplorer, thumbnail
from theme import STYLE


# --------------------------------------------------------------------------
# Build worker
# --------------------------------------------------------------------------
class BuildWorker(QObject):
    statusChanged = Signal(str)
    finished = Signal(object, str)  # gaussian splat, ply path ("" if none)
    failed = Signal(str, str)

    def __init__(self, workspace: Workspace):
        super().__init__()
        self.workspace = workspace

    def run(self):
        run = None
        try:
            ws = self.workspace

            self.statusChanged.emit("Preparing images...")
            run = ws.begin_run()

            images_dir = ws.run_images_dir(run)
            output = ws.run_output_dir(run)

            self.statusChanged.emit("Running InstantSplat Initialiser...")
            inst = InstaSplatBoi(images_dir, output)
            inst.ClearData()
            inst.Build()

            self.statusChanged.emit("Initializing Gaussians...")
            gaussianSplat = GaussianSplat(
                instInput=inst,
                image_path=images_dir,
                output_path=output / "gsplat",
                device="cuda",
                sh_degree=2,
            )

            self.statusChanged.emit("Training Gaussians...")
            gaussianSplat.Train(iterations=1000)

            # images leave the inbox and now live inside the run folder
            ply = ws.complete_run(run)
            self.finished.emit(gaussianSplat, str(ply) if ply else "")

        except Exception as e:
            tb = traceback.format_exc()
            print("\n========== BUILD ERROR ==========")
            print(type(e).__name__)
            print(e)
            print(tb)
            print("=================================\n")

            if run is not None:
                self.workspace.fail_run(run)
            self.failed.emit(str(e), tb)


# --------------------------------------------------------------------------
# Inbox list (drag & drop + delete key)
# --------------------------------------------------------------------------
class InboxList(QListWidget):
    filesDropped = Signal(list)
    deleteRequested = Signal()

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e):
        paths = [
            Path(u.toLocalFile())
            for u in e.mimeData().urls()
            if u.isLocalFile()
        ]
        self.filesDropped.emit(paths)
        e.acceptProposedAction()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Delete:
            self.deleteRequested.emit()
        else:
            super().keyPressEvent(e)


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Forensic Digital Twin")
        self.resize(1500, 900)

        self.manager = WorkspaceManager()
        self.workspace = None
        self.gaussianSplat = None
        self.splatCache = {}  # {workspace name: splat} (only the current one)

        self.buildThread = None
        self.worker = None

        self.buildUi()
        self.refreshWorkspaces()

    # ------------------------------------------------------------- UI
    def buildUi(self):
        self.setDockNestingEnabled(True)
        # bottom dock spans the full window width, Unity style
        self.setCorner(
            Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setCorner(
            Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # ---- toolbar
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(tb)

        brand = QLabel("FORENSIC  TWIN")
        brand.setObjectName("brand")
        tb.addWidget(brand)
        tb.addSeparator()

        wsLabel = QLabel("Workspace")
        wsLabel.setObjectName("muted")
        tb.addWidget(wsLabel)

        self.workspaceCombo = QComboBox()
        self.workspaceCombo.setMinimumWidth(220)
        self.workspaceCombo.currentTextChanged.connect(self.onWorkspaceChosen)
        tb.addWidget(self.workspaceCombo)

        self.newWorkspaceButton = QPushButton("+ New")
        self.newWorkspaceButton.clicked.connect(self.newWorkspace)
        tb.addWidget(self.newWorkspaceButton)

        spacer = QWidget()
        spacer.setObjectName("spacer")
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        tb.addWidget(spacer)

        self.loadPlyButton = QPushButton("Load .ply")
        self.loadPlyButton.clicked.connect(self.loadPly)
        tb.addWidget(self.loadPlyButton)

        # ---- viewport (center)
        self.viewport = Viewport()
        self.setCentralWidget(self.viewport)

        # ---- left dock: image inbox
        panel = QWidget()
        panel.setObjectName("panel")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        self.folderLabel = QLabel("No workspace")
        self.folderLabel.setObjectName("muted")
        self.folderLabel.setWordWrap(True)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.addImagesButton = QPushButton("Add Images")
        self.addImagesButton.clicked.connect(self.addImages)
        self.addFolderButton = QPushButton("Add Folder")
        self.addFolderButton.clicked.connect(self.addFolder)
        row.addWidget(self.addImagesButton)
        row.addWidget(self.addFolderButton)

        self.imageList = InboxList()
        self.imageList.setIconSize(QSize(44, 44))
        self.imageList.setSpacing(2)
        self.imageList.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.imageList.filesDropped.connect(self.addPaths)
        self.imageList.deleteRequested.connect(self.removeSelected)

        self.queueLabel = QLabel("")
        self.queueLabel.setObjectName("muted")
        self.queueLabel.setWordWrap(True)

        self.removeButton = QPushButton("Remove Selected")
        self.removeButton.clicked.connect(self.removeSelected)

        self.buildButton = QPushButton("BUILD SCENE")
        self.buildButton.setObjectName("primary")
        self.buildButton.clicked.connect(self.buildScene)

        lay.addWidget(self.folderLabel)
        lay.addLayout(row)
        lay.addWidget(self.imageList, 1)
        lay.addWidget(self.queueLabel)
        lay.addWidget(self.removeButton)
        lay.addWidget(self.buildButton)

        leftDock = QDockWidget("SCENE IMAGES", self)
        leftDock.setObjectName("imagesDock")
        leftDock.setWidget(panel)
        leftDock.setMinimumWidth(250)
        leftDock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, leftDock)

        # ---- bottom dock: project explorer
        self.explorer = ProjectExplorer()
        self.explorer.plyActivated.connect(self.loadPlyPath)

        bottomDock = QDockWidget("PROJECT", self)
        bottomDock.setObjectName("projectDock")
        bottomDock.setWidget(self.explorer)
        bottomDock.setMinimumHeight(150)
        bottomDock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bottomDock)

        self.resizeDocks([leftDock], [320], Qt.Orientation.Horizontal)
        self.resizeDocks([bottomDock], [260], Qt.Orientation.Vertical)

        # ---- status bar
        self.statusLabel = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(180)
        self.progress.hide()
        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().addWidget(self.statusLabel, 1)
        self.statusBar().addPermanentWidget(self.progress)

    # ------------------------------------------------------ workspaces
    def refreshWorkspaces(self, select=None):
        names = self.manager.list_names()

        self.workspaceCombo.blockSignals(True)
        self.workspaceCombo.clear()
        self.workspaceCombo.addItems(names)
        if select in names:
            self.workspaceCombo.setCurrentText(select)
        self.workspaceCombo.blockSignals(False)

        if names:
            self.openWorkspace(self.workspaceCombo.currentText())
        else:
            self.openWorkspace(None)

    def newWorkspace(self):
        name, ok = QInputDialog.getText(
            self, "New Workspace", "Workspace name:"
        )
        if not ok or not name.strip():
            return

        try:
            ws = self.manager.create(name)
        except ValueError as e:
            QMessageBox.warning(self, "Cannot Create Workspace", str(e))
            return

        self.refreshWorkspaces(select=ws.name)
        self.statusLabel.setText(f'Created workspace "{ws.name}"')

    def onWorkspaceChosen(self, name):
        if name:
            self.openWorkspace(name)

    def openWorkspace(self, name):
        if name is None:
            self.workspace = None
            self.folderLabel.setText("Create a workspace to get started")
            self.explorer.setRoot(None)
            self.showSplat(None)
            self.refreshInbox()
            return

        ws = self.manager.open(name)
        self.workspace = ws
        self.folderLabel.setText(str(ws.root))
        self.explorer.setRoot(ws.project_dir)
        self.refreshInbox()

        # restore this workspace's reconstruction
        splat = self.splatCache.get(ws.name)
        if splat is None:
            ply = ws.latest_ply()
            if ply is not None:
                self.statusLabel.setText(f"Restoring {ply.name}...")
                QApplication.processEvents()
                try:
                    splat = GaussianSplat.FromPLY(str(ply), device="cuda")
                    self.splatCache = {ws.name: splat}
                except Exception:
                    traceback.print_exc()
                    splat = None
                    self.statusLabel.setText("Could not restore reconstruction")

        self.showSplat(splat)
        if splat is not None:
            self.statusLabel.setText(f'Workspace "{ws.name}" loaded')
        elif ws.latest_ply() is None:
            self.statusLabel.setText(f'Workspace "{ws.name}" ready')

    def showSplat(self, splat):
        self.gaussianSplat = splat
        try:
            self.viewport.setSplat(splat)
        except Exception:
            # Viewport may not accept None; only matters when clearing
            if splat is not None:
                raise

    # ----------------------------------------------------------- inbox
    def refreshInbox(self):
        self.imageList.clear()
        count = 0

        if self.workspace is not None:
            for p in self.workspace.inbox_images():
                item = QListWidgetItem(QIcon(thumbnail(p)), p.name)
                self.imageList.addItem(item)
                count += 1

        if self.workspace is None:
            self.queueLabel.setText("")
        elif count == 0:
            self.queueLabel.setText(
                "Drop images here or use Add. Once built, images move to the "
                "Project panel."
            )
        else:
            self.queueLabel.setText(f"{count} images queued")

        self.updateEnabled()

    def addImages(self):
        if self.workspace is None:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Images",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        self.addPaths([Path(f) for f in files])

    def addFolder(self):
        if self.workspace is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Add Image Folder")
        if folder:
            self.addPaths([Path(folder)])

    def addPaths(self, paths):
        if self.workspace is None or self.buildThread is not None:
            return

        files = []
        for p in map(Path, paths):
            if p.is_dir():
                files += [
                    c for c in sorted(p.iterdir())
                    if c.suffix.lower() in IMAGE_EXTS
                ]
            else:
                files.append(p)

        added = self.workspace.add_images(files)
        self.refreshInbox()
        self.statusLabel.setText(f"Added {added} images")

    def removeSelected(self):
        if self.workspace is None or self.buildThread is not None:
            return
        names = [i.text() for i in self.imageList.selectedItems()]
        if not names:
            return
        self.workspace.remove_inbox_images(names)
        self.refreshInbox()

    def updateEnabled(self):
        busy = self.buildThread is not None
        has_ws = self.workspace is not None
        has_imgs = has_ws and self.imageList.count() > 0

        self.workspaceCombo.setEnabled(not busy)
        self.newWorkspaceButton.setEnabled(not busy)
        self.loadPlyButton.setEnabled(not busy)
        self.addImagesButton.setEnabled(has_ws and not busy)
        self.addFolderButton.setEnabled(has_ws and not busy)
        self.removeButton.setEnabled(has_imgs and not busy)
        self.buildButton.setEnabled(has_imgs and not busy)

    # ----------------------------------------------------------- build
    def buildScene(self):
        if self.workspace is None or self.imageList.count() == 0:
            QMessageBox.warning(
                self, "No Images", "Add some images to this workspace first."
            )
            return

        self.buildThread = QThread()
        self.worker = BuildWorker(self.workspace)
        self.worker.moveToThread(self.buildThread)

        self.updateEnabled()
        self.progress.show()
        self.statusLabel.setText("Starting build...")

        self.buildThread.started.connect(self.worker.run)
        self.worker.statusChanged.connect(self.statusLabel.setText)
        self.worker.finished.connect(self.onBuildFinished)
        self.worker.failed.connect(self.onBuildFailed)

        self.worker.finished.connect(self.buildThread.quit)
        self.worker.failed.connect(self.buildThread.quit)
        self.buildThread.finished.connect(self.cleanupThread)

        self.buildThread.start()

    def onBuildFinished(self, gaussianSplat, plyPath):
        self.splatCache = {self.workspace.name: gaussianSplat}
        self.showSplat(gaussianSplat)

        if plyPath:
            self.explorer.reveal(plyPath)
        elif self.workspace.runs:
            self.explorer.navigate(
                self.workspace.run_dir(self.workspace.runs[-1])
            )

        self.statusLabel.setText("Gaussian scene ready")

    def onBuildFailed(self, message, tb):
        self.statusLabel.setText("Build failed")
        QMessageBox.critical(
            self,
            "Build Failed",
            f"{message}\n\nSee console for full traceback.",
        )

    def cleanupThread(self):
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        if self.buildThread is not None:
            self.buildThread.deleteLater()
            self.buildThread = None

        self.progress.hide()
        self.refreshInbox()  # consumed images disappear from the queue

    # ------------------------------------------------------------- ply
    def loadPly(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Gaussian .ply File", "", "PLY Files (*.ply)"
        )
        if path:
            self.loadPlyPath(path)

    def loadPlyPath(self, path):
        if self.buildThread is not None:
            return

        self.statusLabel.setText(f"Loading {Path(path).name}...")
        QApplication.processEvents()

        try:
            splat = GaussianSplat.FromPLY(path, device="cuda")
            self.showSplat(splat)
            self.statusLabel.setText(f"Loaded {Path(path).name}")
        except Exception as e:
            traceback.print_exc()
            QMessageBox.critical(
                self, "Load Failed", f"Could not load .ply file:\n{e}"
            )
            self.statusLabel.setText("Load failed")

    # ----------------------------------------------------------- close
    def closeEvent(self, event):
        if self.buildThread is not None:
            QMessageBox.information(
                self,
                "Build Running",
                "A build is still running. Wait for it to finish before closing.",
            )
            event.ignore()
            return
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
