
from abstractors.projectitem import ProjectItem, ItemType
from abstractors.workers import FolderBuildWorker
class FolderItem(ProjectItem):

    def __init__(self, name, path, parent=None):
        super().__init__(
            name, ItemType.folder, path, parent
        )
    def select(self):
        return FolderBuildWorker(self.path)
        


class ImageItem(ProjectItem):

    def __init__(self, path, parent=None):
        super().__init__(
            path.name, ItemType.image, path, parent
        )

    def select(self, viewport):
        viewport.show_image(self.path)


class PlyItem(ProjectItem):

    def __init__(self, path, parent=None):
        super().__init__(
            path.name, ItemType.ply, path, parent
        )

    def select(self, viewport):
        viewport.load_ply(self.path)


class SceneItem(ProjectItem):

    def __init__(self, name, splat, parent=None):
        super().__init__(
            name, ItemType.scene, parent=parent
        )

        self.splat = splat

    def select(self, viewport):
        viewport.show_splat(
            self.splat,
            self.name,
            f"scene:{id(self)}"
        )
