import sys
from pathlib import Path
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStackedWidget
)
from theme import STYLE

from App_Elements.workspaceStuff import WorkspaceSelect
from App_Elements.mainStuff import SceneBuilder

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ForensiC Digital Twin")
        self.resize(1400, 850)
        self.workspacePath = ""
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.workspacePage = WorkspaceSelect()
        self.stack.addWidget(self.workspacePage)

        self.mainPage = SceneBuilder(self.workspacePath)
        self.stack.addWidget(self.mainPage)

        self.stack.setCurrentWidget(self.workspacePage)
        self.workspacePage.workspace_opened.connect(self.openWorkspace)

    def openWorkspace(self, workspace):
        self.stack.setCurrentWidget(self.mainPage)
        self.mainPage.SetWorkspacePath(workspace["path"])



def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()