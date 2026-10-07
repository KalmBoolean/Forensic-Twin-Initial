from pathlib import Path
import shutil
from instantsplat.initializer import Dust3rInitializer
import os
from instantsplat.initializer import TrainableInitializedCameraDataset
from gaussian_splatting import CameraTrainableGaussianModel
from instantsplat.initializer import TrainableInitializedCameraDataset

class InstaSplatBoi():

    def __init__(self,imgPath,outPath):
        
        self.image_path_list = [os.path.join(imgPath, file) for file in sorted(os.listdir(imgPath))]
        self.initializer = Dust3rInitializer().to("cuda")
        self.output = Path(outPath)
        self.gaussians = CameraTrainableGaussianModel(sh_degree=2).to("cuda")

    def Build(self):
        initialized_point_cloud, initialized_cameras = self.initializer(image_path_list=self.image_path_list)
        self.cameras = initialized_cameras[:]
        self.dataset = TrainableInitializedCameraDataset(initialized_cameras).to("cuda")   
        self.gaussians.create_from_pcd(initialized_point_cloud.points, initialized_point_cloud.colors)
    
    def ClearData(self):
        print("nothing to clear :p")   