
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QMessageBox,
)

from .viewport import Viewport
from gaussplat import GaussianSplat


class ImageViewer(QScrollArea):

    def __init__(self, image_path, parent=None):
        super().__init__(parent)

        self.image_path = Path(image_path)

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Ignored
        )

        self.setWidget(self.image)
        self.setWidgetResizable(True)

        self.pixmap = QPixmap(str(self.image_path))

        self.image.setPixmap(self.pixmap)
        self.image.resize(self.pixmap.size())


class ViewportArea(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.open_tabs = {}
        self.k = 0
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)

        self.tabs.tabCloseRequested.connect(
            self.close_tab
        )

        layout.addWidget(self.tabs)

    def open_tab(self, key, name, widget):

        if key in self.open_tabs:
            self.tabs.setCurrentWidget(
                self.open_tabs[key]
            )
            return self.open_tabs[key]

        self.open_tabs[key] = widget

        index = self.tabs.addTab(widget, name)
        self.tabs.setCurrentIndex(index)

        return widget

    def close_tab(self, index):
        widget = self.tabs.widget(index)

        if widget is None:
            return
        
        for key, value in list(self.open_tabs.items()):
            if value is widget:
                del self.open_tabs[key]
                break

        self.tabs.removeTab(index)
        widget.deleteLater()


    def show_image(self, path):
        path = Path(path)
        key = f"image:{path.resolve()}"

        if key in self.open_tabs:
            self.tabs.setCurrentWidget(
                self.open_tabs[key]
            )
            return

        viewer = ImageViewer(path)

        if viewer.pixmap.isNull():
            QMessageBox.warning(
                self,
                "Image Error",
                f"Could not load image:\n{path}"
            )
            viewer.deleteLater()
            return

        self.open_tab(
            key,
            path.name,
            viewer
        )

    def show_splat(self, splat, name):
        key = f"splat:{self.k}"
        self.k += 1
        if key in self.open_tabs:
            self.tabs.setCurrentWidget(
                self.open_tabs[key]
            )
            return

        viewport = Viewport()
        viewport.setSplat(splat)

        self.open_tab(
            key,
            name,
            viewport
        )

    def load_ply(self, path):
        path = Path(path)
        key = f"ply:{path.resolve()}"

        if key in self.open_tabs:
            self.tabs.setCurrentWidget(
                self.open_tabs[key]
            )
            return

        try:
            splat = GaussianSplat.FromPLY(
                str(path),
                device="cuda"
            )

            viewport = Viewport()
            viewport.setSplat(splat)

            self.open_tab(
                key,
                path.name,
                viewport
            )

        except Exception as e:
            import traceback
            traceback.print_exc()

            QMessageBox.critical(
                self,
                "PLY Loading Failed",
                str(e)
            )
