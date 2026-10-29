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
        self.roll = 0.0
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

        forward = np.array(
            [sy * cp, -sp, -cy * cp],
            dtype=np.float32,
        )
        right = np.array(
            [cy, 0.0, sy],
            dtype=np.float32,
        )
        up = np.cross(right, forward)

        rotation = np.stack([right, up, -forward], axis=0)

        viewmat = np.eye(4, dtype=np.float32)
        viewmat[:3, :3] = rotation
        viewmat[:3, 3] = -rotation @ self.position

        return torch.tensor(
            viewmat,
            dtype=torch.float32,
            device=self.device,
        )

    def view_matrix_with_roll(self):
        """Apply camera roll without changing the camera's world position."""
        viewmat = self.view_matrix().clone()

        roll = np.radians(self.roll)
        cr, sr = np.cos(roll), np.sin(roll)

        R_roll = torch.tensor(
            [
                [cr, -sr, 0.0],
                [sr,  cr, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=viewmat.dtype,
            device=viewmat.device,
        )

        original_rotation = viewmat[:3, :3].clone()
        original_translation = viewmat[:3, 3].clone()

        # Recover the camera's world-space position.
        camera_center = -(
            original_rotation.transpose(0, 1)
            @ original_translation
        )

        # Rotate the camera orientation around its local viewing axis.
        rolled_rotation = R_roll @ original_rotation

        # Recalculate translation to preserve the camera center.
        viewmat[:3, :3] = rolled_rotation
        viewmat[:3, 3] = -(rolled_rotation @ camera_center)

        return viewmat.contiguous()

    def rotate_roll(self, delta):
        self.roll += delta * self.mouseSensitivity

    def pan(self, dx, dy):
        """Pan relative to the camera's rolled image plane."""
        yaw = np.radians(self.yaw)
        pitch = np.radians(self.pitch)
        roll = np.radians(self.roll)

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)

        forward = np.array(
            [sy * cp, -sp, -cy * cp],
            dtype=np.float32,
        )
        right = np.array(
            [cy, 0.0, sy],
            dtype=np.float32,
        )
        up = np.cross(right, forward)

        cr, sr = np.cos(roll), np.sin(roll)

        rolled_right = cr * right - sr * up
        rolled_up = sr * right + cr * up

        self.position += (
            rolled_right * float(dx)
            + rolled_up * float(dy)
        ).astype(np.float32)

    # ---- Movement ---------------------------------------------------

    def move(self, x, y, z):
        """Move without allowing diagonal input to increase speed."""
        direction = np.array([x, y, z], dtype=np.float32)

        magnitude = float(np.linalg.norm(direction))
        if magnitude > 1.0:
            direction /= magnitude

        yaw = np.radians(self.yaw)
        pitch = np.radians(self.pitch)

        forward = np.array(
            [
                np.sin(yaw) * np.cos(pitch),
                -np.sin(pitch),
                -np.cos(yaw) * np.cos(pitch),
            ],
            dtype=np.float32,
        )

        right = np.array(
            [np.cos(yaw), 0.0, np.sin(yaw)],
            dtype=np.float32,
        )

        world_up = np.array(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        )

        # Roll does not affect movement speed or forward direction.
        self.position += (
            right * direction[0]
            + world_up * direction[1]
            + forward * direction[2]
        )

    def rotate(self, dx, dy):
        """Rotate the camera without modifying its position."""
        self.yaw += dx * self.mouseSensitivity
        self.pitch += dy * self.mouseSensitivity

        self.pitch = max(-89.0, min(89.0, self.pitch))

    def adjustSpeed(self, factor):
        self.moveSpeed = max(
            0.001,
            min(2.0, self.moveSpeed * factor),
        )

    # ---- Framing ----------------------------------------------------

    def frame(self, means):
        if means.shape[0] == 0:
            return

        center = means.mean(axis=0)
        extent = float(np.linalg.norm(means.std(axis=0)))
        distance = max(extent * 3.0, 0.5)

        self.position = (
            center
            + np.array([0.0, 0.0, distance], dtype=np.float32)
        )

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        self.moveSpeed = max(distance * 0.02, 0.001)

    def reset(self):
        self.position = np.array(
            [0.0, 0.0, 2.0],
            dtype=np.float32,
        )

        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.moveSpeed = 0.05


