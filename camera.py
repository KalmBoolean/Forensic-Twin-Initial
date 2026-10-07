import numpy as np
import torch


class Camera:

    def __init__(
        self,
        device,
        position=(0.0, 0.0, 2.0),
        yaw=0.0,
        pitch=0.0,
        fov_degrees=60.0,
        width=800,
        height=600,
    ):
        self.device = device

        self.position = np.array(position, dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch

        self.fovDegrees = fov_degrees
        self.width = max(int(width), 64)
        self.height = max(int(height), 64)

        self.moveSpeed = 0.05
        self.mouseSensitivity = 0.25

    def resize(self, width, height):
        self.width = max(int(width), 64)
        self.height = max(int(height), 64)

    @property
    def image_width(self):
        return self.width

    @property
    def image_height(self):
        return self.height

    @property
    def K(self):
        fov = np.radians(self.fovDegrees)
        focal = self.height / (2.0 * np.tan(fov / 2.0))

        return torch.tensor(
            [
                [focal, 0.0, self.width / 2.0],
                [0.0, focal, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def view_matrix(self):
        yaw = np.radians(self.yaw)
        pitch = np.radians(self.pitch)

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)

        forward = np.array([sy * cp, -sp, -cy * cp], dtype=np.float32)
        right = np.array([cy, 0.0, sy], dtype=np.float32)
        up = np.cross(right, forward)

        rotation = np.stack([right, up, -forward], axis=0)

        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = rotation
        viewmat[:3, 3] = -rotation @ self.position

        return torch.tensor(viewmat, dtype=torch.float32, device=self.device)

    # ---- movement ---------------------------------------------------

    def move(self, x, y, z):
        yaw = np.radians(self.yaw)

        forward = np.array([np.sin(yaw), 0.0, -np.cos(yaw)], dtype=np.float32)
        right = np.array([np.cos(yaw), 0.0, np.sin(yaw)], dtype=np.float32)

        self.position += (
            right * x
            + np.array([0.0, 1.0, 0.0], dtype=np.float32) * y
            + forward * z
        )

    def rotate(self, dx, dy):
        self.yaw += dx * self.mouseSensitivity
        self.pitch += dy * self.mouseSensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def adjustSpeed(self, factor):
        self.moveSpeed = max(0.001, min(2.0, self.moveSpeed * factor))

    # ---- framing ------------------------------------------------------

    def frame(self, means):

        if means.shape[0] == 0:
            return

        center = means.mean(axis=0)
        extent = float(np.linalg.norm(means.std(axis=0)))
        distance = max(extent * 3.0, 0.5)

        self.position = center + np.array([0.0, 0.0, distance], dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0
        self.moveSpeed = max(distance * 0.02, 0.001)

    def reset(self):
        self.position = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0