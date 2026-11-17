
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QTreeWidget,
    QLabel,
    QTreeWidgetItem,
)

from abstractors.projectitem import ProjectItem


class LeftPanel(QWidget):

    def __init__(
        self,
        viewport,
        on_new_build,
        parent=None
    ):
        super().__init__(parent)

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

        if isinstance(item, ProjectItem):
            item.select(self.viewport)

    def clear_project(self):
        self.root.takeChildren()
