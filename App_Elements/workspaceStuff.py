import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

savePath = Path.home() / ".workspace_app" / "workspaces.json"


def LoadWorkspaces():
    try:
        return json.loads(savePath.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def SaveWorkspace(workspaces):
    savePath.parent.mkdir(parents=True, exist_ok=True)
    savePath.write_text(json.dumps(workspaces, indent=2))


class WorkspaceSelect(QWidget):
    
    workspace_opened = Signal(dict)

    def __init__(self):
        super().__init__()

        title = QLabel("Select a workspace")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(self.OpenSelectedWorkspace)
        self.list.itemSelectionChanged.connect(self.UpdateUI)

        self.new_button = QPushButton("New workspace...")
        self.new_button.clicked.connect(self.CreateWorkspace)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.RemoveSelected)
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.OpenSelectedWorkspace)

        buttons = QHBoxLayout()
        buttons.addWidget(self.new_button)
        buttons.addStretch()
        buttons.addWidget(self.remove_button)
        buttons.addWidget(self.open_button)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.list)
        layout.addLayout(buttons)

        self.Refresh()

    def Refresh(self):
        self.list.clear()
        for workspace in LoadWorkspaces():
            item = QListWidgetItem(workspace["name"] + "\n" + workspace["path"])
            item.setData(Qt.UserRole, workspace)
            self.list.addItem(item)
        self.UpdateUI()

    def GetSelectedWorkspaceData(self):
        item = self.list.currentItem()
        if item is None:
            return None
        return item.data(Qt.UserRole)

    def UpdateUI(self):
        isSelected = self.GetSelectedWorkspaceData() is not None
        self.open_button.setEnabled(isSelected)
        self.remove_button.setEnabled(isSelected)

    def OpenSelectedWorkspace(self):
        workspace = self.GetSelectedWorkspaceData()
        if workspace:
            self.workspace_opened.emit(workspace)

    def CreateWorkspace(self):
        name, ok = QInputDialog.getText(self, "New workspace", "Workspace name:")
        name = name.strip()
        if not ok or not name:
            return

        parent_folder = QFileDialog.getExistingDirectory(self, "Where should it live?")
        if not parent_folder:
            return

        path = Path(parent_folder) / name
        try:
            path.mkdir(parents=True)
        except OSError as error:
            QMessageBox.warning(self, "Couldn't create workspace", str(error))
            return

        workspace = {"name": name, "path": str(path)}
        workspaces = LoadWorkspaces()
        workspaces.append(workspace)
        SaveWorkspace(workspaces)

        self.Refresh()
        self.workspace_opened.emit(workspace)

    def RemoveSelected(self):
        workspace = self.GetSelectedWorkspaceData()
        if workspace is None:
            return
        workspaces = [w for w in LoadWorkspaces() if w["path"] != workspace["path"]]
        SaveWorkspace(workspaces)
        self.Refresh()