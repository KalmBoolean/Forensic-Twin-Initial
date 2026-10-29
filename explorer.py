"""Unity-style project browser: folder tree on the left, thumbnail grid on the right."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QDir, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QImageReader,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListView,
    QSlider,
    QSplitter,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

try:  # location differs between PySide6 versions
    from PySide6.QtGui import QFileSystemModel
except ImportError:
    from PySide6.QtWidgets import QFileSystemModel

from workspace import IMAGE_EXTS

_THUMB_SIZE = 128
_thumb_cache: dict[str, QPixmap] = {}
_icon_cache: dict[str, QIcon] = {}


def thumbnail(path: Path | str, size: int = _THUMB_SIZE) -> QPixmap:
    """Fast, cached, aspect-preserving thumbnail (decodes at reduced size)."""
    key = f"{path}|{size}"
    if key in _thumb_cache:
        return _thumb_cache[key]

    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    src = reader.size()
    if src.isValid():
        reader.setScaledSize(
            src.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio)
        )
    img = reader.read()
    pm = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
    _thumb_cache[key] = pm
    return pm


def _badge_icon(kind: str, label: str = "", color: str = "#19d3ff") -> QIcon:
    key = f"{kind}|{label}|{color}"
    if key in _icon_cache:
        return _icon_cache[key]

    pm = QPixmap(128, 128)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)

    if kind == "folder":
        p.setBrush(QColor("#2a2a33"))
        p.drawRoundedRect(10, 24, 50, 26, 7, 7)
        p.setBrush(QColor("#34343f"))
        p.drawRoundedRect(10, 36, 108, 72, 10, 10)
        p.setBrush(QColor(color))
        p.drawRoundedRect(10, 36, 108, 6, 3, 3)
    else:
        p.setBrush(QColor("#17171c"))
        p.setPen(QPen(QColor(color), 3))
        p.drawRoundedRect(28, 10, 72, 108, 10, 10)
        p.setPen(QColor(color))
        font = QFont("Segoe UI", 20)
        font.setBold(True)
        p.setFont(font)
        p.drawText(pm.rect(), Qt.AlignmentFlag.AlignCenter, label)

    p.end()
    icon = QIcon(pm)
    _icon_cache[key] = icon
    return icon


class ThumbModel(QFileSystemModel):
    """File system model that shows image thumbnails and custom flat icons."""

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DecorationRole and index.column() == 0:
            if self.isDir(index):
                return _badge_icon("folder")

            path = Path(self.filePath(index))
            suffix = path.suffix.lower()

            if suffix in IMAGE_EXTS:
                pm = thumbnail(path)
                if not pm.isNull():
                    return QIcon(pm)
            if suffix == ".ply":
                return _badge_icon("file", "PLY", "#a78bfa")

            return _badge_icon(
                "file", suffix.lstrip(".").upper()[:4], "#7a7a85"
            )

        return super().data(index, role)


class ProjectExplorer(QWidget):
    plyActivated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.root: Path | None = None
        self.current: Path | None = None

        self.dirModel = ThumbModel(self)
        self.dirModel.setFilter(
            QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot
        )
        self.fileModel = ThumbModel(self)
        self.fileModel.setFilter(
            QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot
        )

        # ---- header: up / breadcrumb / zoom
        self.upButton = QToolButton()
        self.upButton.setObjectName("flat")
        self.upButton.setText("\u2190")
        self.upButton.setToolTip("Up one folder")
        self.upButton.clicked.connect(self.goUp)

        self.rootButton = QToolButton()
        self.rootButton.setObjectName("flat")
        self.rootButton.setText("Project")
        self.rootButton.clicked.connect(lambda: self.root and self.navigate(self.root))

        self.crumb = QLabel("")
        self.crumb.setObjectName("muted")

        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(48, 192)
        self.zoom.setValue(96)
        self.zoom.setFixedWidth(120)
        self.zoom.valueChanged.connect(self.applyZoom)

        header = QHBoxLayout()
        header.setContentsMargins(8, 6, 12, 6)
        header.setSpacing(4)
        header.addWidget(self.upButton)
        header.addWidget(self.rootButton)
        header.addWidget(self.crumb, 1)
        header.addWidget(self.zoom)

        # ---- folder tree
        self.tree = QTreeView()
        self.tree.setModel(self.dirModel)
        self.tree.setHeaderHidden(True)
        self.tree.setIconSize(QSize(18, 18))
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for col in (1, 2, 3):
            self.tree.hideColumn(col)
        self.tree.clicked.connect(
            lambda idx: self.navigate(self.dirModel.filePath(idx))
        )

        # ---- thumbnail grid
        self.list = QListView()
        self.list.setModel(self.fileModel)
        self.list.setViewMode(QListView.ViewMode.IconMode)
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setMovement(QListView.Movement.Static)
        self.list.setWordWrap(True)
        self.list.setUniformItemSizes(True)
        self.list.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.list.setSpacing(4)
        self.list.doubleClicked.connect(self.onActivated)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.list)
        splitter.setSizes([220, 900])
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addLayout(header)
        layout.addWidget(splitter, 1)

        self.applyZoom(self.zoom.value())
        self.setEnabled(False)

    # ------------------------------------------------------------- API
    def setRoot(self, path: Path | str | None):
        if path is None:
            self.root = self.current = None
            self.crumb.setText("")
            self.setEnabled(False)
            return

        self.setEnabled(True)
        self.root = Path(path)
        self.dirModel.setRootPath(str(self.root))
        self.fileModel.setRootPath(str(self.root))
        self.tree.setRootIndex(self.dirModel.index(str(self.root)))
        self.navigate(self.root)

    def navigate(self, path: Path | str):
        if self.root is None:
            return

        path = Path(path)
        if not path.is_dir():
            path = self.root
        self.current = path

        self.list.setRootIndex(self.fileModel.index(str(path)))

        if path == self.root:
            self.tree.clearSelection()
        else:
            self.tree.setCurrentIndex(self.dirModel.index(str(path)))

        rel = path.relative_to(self.root).parts if path != self.root else ()
        self.crumb.setText("  /  ".join(rel))
        self.upButton.setEnabled(path != self.root)

    def reveal(self, file_path: Path | str):
        """Jump to the folder containing file_path and select the file."""
        file_path = Path(file_path)
        self.navigate(file_path.parent)
        idx = self.fileModel.index(str(file_path))
        if idx.isValid():
            self.list.setCurrentIndex(idx)

    # ------------------------------------------------------- internals
    def goUp(self):
        if self.current and self.root and self.current != self.root:
            self.navigate(self.current.parent)

    def applyZoom(self, size: int):
        self.list.setIconSize(QSize(size, size))
        self.list.setGridSize(QSize(size + 28, size + 44))

    def onActivated(self, index):
        path = Path(self.fileModel.filePath(index))
        if path.is_dir():
            self.navigate(path)
        elif path.suffix.lower() == ".ply":
            self.plyActivated.emit(str(path))
