from pathlib import Path
from enum import Enum, auto
from PySide6.QtWidgets import QTreeWidgetItem


class ItemType(Enum):
    build = auto()
    folder = auto()
    image = auto()
    ply = auto()
    scene = auto()


class ProjectItem(QTreeWidgetItem):

    def __init__(self, name, item_type, path=None, parent=None):
        super().__init__(parent, [name])

        self.name = name
        self.item_type = item_type
        self.path = Path(path) if path else None

        self.children = []

        if isinstance(parent, ProjectItem):
            parent.children.append(self)

    def select(self, viewport):
        pass
