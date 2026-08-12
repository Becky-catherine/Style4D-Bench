#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.dataset import FourDGSdataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from torch.utils.data import Dataset
from scene.dataset_readers import add_points
class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False, initialize_gaussians=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.args = args
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.video_cameras = {}
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, args.llffhold)
            dataset_type="colmap"
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args.extension)
            dataset_type="blender"
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](args.source_path, args.white_background, args.eval)
            dataset_type="dynerf"
        elif os.path.exists(os.path.join(args.source_path,"dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](args.source_path, False, args.eval)
            dataset_type="nerfies"
        elif os.path.exists(os.path.join(args.source_path,"train_meta.json")):
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](args.source_path)
            dataset_type="PanopticSports"
        elif os.path.exists(os.path.join(args.source_path,"points3D_multipleview.ply")):
            scene_info = sceneLoadTypeCallbacks["MultipleView"](args.source_path)
            dataset_type="MultipleView"
        else:
            assert False, "Could not recognize scene type!"
        self.maxtime = scene_info.maxtime
        self.dataset_type = dataset_type
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, dataset_type)

        # self.video_camera = cameraList_from_camInfos(scene_info.video_cameras,-1,args)
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        if args.add_points:
            print("add points.")
            # breakpoint()
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        self.gaussians._deformation.deformation_net.set_aabb(xyz_max,xyz_min)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_model(os.path.join(self.model_path,
                                                    "point_cloud",
                                                    "iteration_" + str(self.loaded_iter),
                                                   ))
        elif initialize_gaussians:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)
        else:
            # Stage two immediately loads the first-stage point cloud. Avoid a
            # redundant GPU allocation for a point cloud that would be replaced.
            self.gaussians.spatial_lr_scale = self.cameras_extent

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_deformation(point_cloud_path)
    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera
    def getVideoCameras(self, scale=1.0):
        return self.video_camera
    def reload_dataset(self, new_source_path):

        print(f"Reloading dataset from {new_source_path}...")

        # 更新数据集路径
        self.dataset_type = None
        self.cameras_extent = None

        # 重新识别数据集类型
        if os.path.exists(os.path.join(new_source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](new_source_path, None, None, None)
            self.dataset_type = "colmap"
        elif os.path.exists(os.path.join(new_source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](new_source_path, self.args.white_background, self.args.eval, self.args.extension)
            #scene_info = sceneLoadTypeCallbacks["Blender"](new_source_path, False, None, None)
            self.dataset_type = "blender"
        elif os.path.exists(os.path.join(new_source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["dynerf"](new_source_path, False, None)
            self.dataset_type = "dynerf"
        elif os.path.exists(os.path.join(new_source_path, "dataset.json")):
            scene_info = sceneLoadTypeCallbacks["nerfies"](new_source_path, False, None)
            self.dataset_type = "nerfies"
        elif os.path.exists(os.path.join(new_source_path, "train_meta.json")):
            scene_info = sceneLoadTypeCallbacks["PanopticSports"](new_source_path)
            self.dataset_type = "PanopticSports"
        elif os.path.exists(os.path.join(new_source_path, "points3D_multipleview.ply")):
            scene_info = sceneLoadTypeCallbacks["MultipleView"](new_source_path)
            self.dataset_type = "MultipleView"
        else:
            raise ValueError("Could not recognize scene type!")

        # 更新 `self.cameras_extent`
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.maxtime = scene_info.maxtime

        # 仅更新数据，不重新初始化 gaussians
        print("Updating camera datasets...")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, None, self.dataset_type)
        self.test_camera = FourDGSdataset(scene_info.test_cameras, None, self.dataset_type)
        self.video_camera = FourDGSdataset(scene_info.video_cameras, None, self.dataset_type)

        print("Dataset reloaded successfully!")
