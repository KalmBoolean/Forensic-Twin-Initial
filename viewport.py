import numpy as np
import torch

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

from camera import Camera


class Viewport(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.splats = None
        self.camera = Camera(
            device=torch.device("cpu"),
            width=self.width(),
            height=self.height(),
        )

        self.lastMouse = None
        self.mouseCaptured = False

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def setSplat(self, splat):
        self.splats = splat
        self.camera.device = self.splats.device
        self.setFocus()
        self.frameScene()
        self.renderScene()

    def frameScene(self):
        if self.splats is None or self.splats.splats is None:
            return

        means = self.splats.splats["means"].detach().cpu().numpy()
        self.camera.frame(means)

    def renderScene(self):
        if self.splats is None:
            return

        self.camera.resize(self.width(), self.height())

        rendered, _ = self.splats.Render(
            self.camera,
            self.camera.view_matrix()
        )

        image = rendered.detach().cpu().numpy()
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        qimage = QImage(
            image.data,
            image.shape[1],
            image.shape[0],
            image.strides[0],
            QImage.Format.Format_RGB888
        ).copy()

        pixmap = QPixmap.fromImage(qimage)

        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self.splats is not None:
            self.renderScene()

    def keyPressEvent(self, event):
        speed = self.camera.moveSpeed

        if event.key() == Qt.Key.Key_W:
            self.camera.move(0, 0, -speed)
        elif event.key() == Qt.Key.Key_S:
            self.camera.move(0, 0, speed)
        elif event.key() == Qt.Key.Key_A:
            self.camera.move(-speed, 0, 0)
        elif event.key() == Qt.Key.Key_D:
            self.camera.move(speed, 0, 0)
        elif event.key() == Qt.Key.Key_Q:
            self.camera.move(0, speed, 0)
        elif event.key() == Qt.Key.Key_E:
            self.camera.move(0, -speed, 0)

        self.renderScene()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.lastMouse = event.position()
            self.mouseCaptured = True
            self.setCursor(Qt.CursorShape.BlankCursor)

    def mouseMoveEvent(self, event):
        if not self.mouseCaptured:
            return

        current = event.position()
        dx = current.x() - self.lastMouse.x()
        dy = current.y() - self.lastMouse.y()
        self.lastMouse = current

        self.camera.rotate(dx, dy)
        self.renderScene()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouseCaptured = False
            self.lastMouse = None
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.camera.adjustSpeed(1.15 if delta > 0 else 0.87)

    def resetCamera(self):
        self.camera.reset()
        self.renderScene()