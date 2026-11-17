
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QTreeWidget
)
from PySide6.QtWidgets import QTreeWidgetItem

class RightPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        self.searchBar = QLineEdit()
        self.searchBar.setPlaceholderText("Search scene objects...")

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)

        layout.addWidget(self.searchBar)
        layout.addWidget(self.tree)

        self.searchBar.textChanged.connect(self.FilterTree)
        self.tree.itemDoubleClicked.connect(self.OnItemDoubleClicked)

    def AddCategory(self, name, parent=None):

        category = QTreeWidgetItem([name])

        if parent is None:
            self.tree.addTopLevelItem(category)
        else:
            parent.addChild(category)

        return category

    def AddProjectItem(self, projectItem, parent):
        parent.addChild(projectItem)

    def OnItemDoubleClicked(self, item):
        if hasattr(item, "select"):
            item.select()

    def FilterTree(self, text):
        text = text.lower().strip()

        def FilterItem(item):
            childMatches = False

            for i in range(item.childCount()):
                if FilterItem(item.child(i)):
                    childMatches = True

            ownMatch = text in item.text(0).lower()
            visible = ownMatch or childMatches

            item.setHidden(not visible)

            if text and childMatches:
                item.setExpanded(True)

            return visible

        for i in range(self.tree.topLevelItemCount()):
            FilterItem(self.tree.topLevelItem(i))
