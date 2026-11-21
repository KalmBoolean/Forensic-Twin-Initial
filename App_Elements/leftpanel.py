
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTreeWidget,
    QLabel,
    QTreeWidgetItem,
)

from abstractors.projectitem import ProjectItem
from App_Elements.projectitems import FolderItem


class LeftPanel(QWidget):

    def __init__(
        self,
        viewport,
        on_new_build,
        update_status,
        build_finished,
        build_failed,
        threadpool,
        parent=None
    ):
        super().__init__(parent)
        self.threadPool = threadpool
        self.on_update_status = update_status
        self.on_build_finished = build_finished
        self.on_build_failed = build_failed
        self.viewport = viewport
        self.on_new_build = on_new_build
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Project Explorer")
        title.setStyleSheet(
            "font-size: 16px; font-weight: bold;"
        )

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(
            QTreeWidget.SelectionMode.SingleSelection
        )

        self.root = QTreeWidgetItem(
            self.tree, ["Builds"]
        )

        self.new_build_button = QPushButton("+ New Build")
        self.new_build_button.clicked.connect(
            self.on_new_build
        )

        self.select_button = QPushButton("Select")
        self.select_button.clicked.connect(
            self.select_item
        )

        layout.addWidget(title)
        layout.addWidget(self.tree)
        layout.addWidget(self.new_build_button)
        layout.addWidget(self.select_button)

    def add_item(self, parent, item):
        parent.addChild(item)

    def select_item(self):
        item = self.tree.currentItem()

        if not isinstance(item, ProjectItem):
            return

        if isinstance(item, FolderItem):
            worker = item.select()

            worker.signals.statusChanged.connect(
                self.on_update_status
            )
            worker.signals.finished.connect(
                lambda splat: self.on_build_finished(splat)
            )
            worker.signals.failed.connect(
                self.on_build_failed
            )

            self.threadPool.start(worker)

        else:
            item.select(self.viewport)


    def clear_project(self):
        self.root.takeChildren()
