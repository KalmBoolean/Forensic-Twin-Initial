"""Black / sleek QSS theme."""

COLORS = {
    "%BG%": "#09090b",
    "%SURFACE%": "#101013",
    "%RAISED%": "#17171c",
    "%BORDER%": "#23232a",
    "%TEXT%": "#e6e6ea",
    "%MUTED%": "#7a7a85",
    "%ACCENT%": "#19d3ff",
    "%ACCENT_DIM%": "#0f7c96",
}

_QSS = """
* {
    font-family: "Segoe UI", "Inter", "SF Pro Text", "Helvetica Neue", sans-serif;
    font-size: 13px;
    color: %TEXT%;
}

QMainWindow, QDialog, QMessageBox { background: %BG%; }
QWidget#panel { background: %SURFACE%; }
QWidget#spacer { background: transparent; }
QLabel { background: transparent; }
QLabel#brand { color: %ACCENT%; font-weight: 700; letter-spacing: 3px; }
QLabel#muted { color: %MUTED%; font-size: 12px; }

/* ---------- toolbar / status ---------- */
QToolBar {
    background: %SURFACE%;
    border: none;
    border-bottom: 1px solid %BORDER%;
    padding: 6px 10px;
    spacing: 8px;
}
QToolBar::separator { background: %BORDER%; width: 1px; margin: 4px 8px; }

QStatusBar {
    background: %SURFACE%;
    border-top: 1px solid %BORDER%;
    color: %MUTED%;
}
QStatusBar::item { border: none; }
QStatusBar QLabel { color: %MUTED%; padding-left: 6px; }

QProgressBar {
    background: %BORDER%;
    border: none;
    max-height: 3px;
    min-height: 3px;
}
QProgressBar::chunk { background: %ACCENT%; }

/* ---------- buttons ---------- */
QPushButton, QToolButton {
    background: %RAISED%;
    border: 1px solid %BORDER%;
    border-radius: 6px;
    padding: 6px 12px;
}
QPushButton:hover, QToolButton:hover {
    border-color: %ACCENT_DIM%;
    background: #1d1d24;
}
QPushButton:pressed, QToolButton:pressed { background: #0f0f13; }
QPushButton:disabled, QToolButton:disabled {
    color: #4a4a52;
    background: %SURFACE%;
    border-color: %BORDER%;
}

QPushButton#primary {
    background: %ACCENT%;
    color: #031014;
    border: none;
    font-weight: 700;
    padding: 11px 14px;
    letter-spacing: 1px;
}
QPushButton#primary:hover { background: #5fe3ff; }
QPushButton#primary:disabled { background: #14262b; color: #4d6a70; }

QToolButton#flat { border: none; background: transparent; padding: 4px 8px; }
QToolButton#flat:hover { background: %RAISED%; }
QToolButton#flat:disabled { background: transparent; }

/* ---------- inputs ---------- */
QComboBox, QLineEdit {
    background: %RAISED%;
    border: 1px solid %BORDER%;
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: %ACCENT_DIM%;
}
QComboBox:hover, QLineEdit:focus { border-color: %ACCENT_DIM%; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: %RAISED%;
    border: 1px solid %BORDER%;
    selection-background-color: %ACCENT_DIM%;
    outline: none;
}

QSlider::groove:horizontal { height: 3px; background: %BORDER%; border-radius: 1px; }
QSlider::sub-page:horizontal { background: %ACCENT_DIM%; border-radius: 1px; }
QSlider::handle:horizontal {
    background: %ACCENT%;
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
}

/* ---------- docks / splitters ---------- */
QDockWidget { font-weight: 600; }
QDockWidget::title {
    background: %SURFACE%;
    color: %MUTED%;
    padding: 8px 12px;
    border-bottom: 1px solid %BORDER%;
    text-align: left;
}
QMainWindow::separator { background: %BORDER%; width: 1px; height: 1px; }
QMainWindow::separator:hover { background: %ACCENT%; }
QSplitter::handle { background: %BORDER%; }
QSplitter::handle:horizontal { width: 1px; }
QSplitter::handle:hover { background: %ACCENT_DIM%; }

/* ---------- lists / trees ---------- */
QListView, QListWidget, QTreeView {
    background: %SURFACE%;
    border: none;
    outline: none;
    alternate-background-color: %SURFACE%;
}
QListView::item, QListWidget::item { border-radius: 6px; padding: 4px; }
QListView::item:hover, QListWidget::item:hover, QTreeView::item:hover {
    background: %RAISED%;
}
QListView::item:selected, QListWidget::item:selected, QTreeView::item:selected {
    background: rgba(25, 211, 255, 0.16);
    color: white;
}
QTreeView::item { padding: 4px 2px; border-radius: 4px; }

/* ---------- scrollbars ---------- */
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical {
    background: #2a2a32; border-radius: 5px; min-height: 28px; margin: 2px;
}
QScrollBar::handle:vertical:hover { background: %ACCENT_DIM%; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal {
    background: #2a2a32; border-radius: 5px; min-width: 28px; margin: 2px;
}
QScrollBar::handle:horizontal:hover { background: %ACCENT_DIM%; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QToolTip {
    background: %RAISED%;
    color: %TEXT%;
    border: 1px solid %BORDER%;
    padding: 4px 8px;
}
"""


def build_style() -> str:
    qss = _QSS
    for token, value in COLORS.items():
        qss = qss.replace(token, value)
    return qss


STYLE = build_style()
