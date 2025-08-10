import os

import pathlib

import numpy as np
import cv2
import open3d as o3d

from apriltags.detect_apriltags import detect_apriltags
from utils.utils import read_pickle
from utils.images import prepare_imgs, process_images
import pycolmap

################## DATASET SELECTION ##################
DATASETS_PATH = "datasets/stereo_raices"
SEQUENCE_NAME = "raiz_apriltags_camaramovil"

CALIB_STEREO_FILE = os.path.join(DATASETS_PATH, "stereo_calibration.pkl")
UNDISTORT_MAPS_FILE = os.path.join(DATASETS_PATH, "stereo_maps.pkl")

SEQUENCE_PATH = os.path.join(DATASETS_PATH, "captures", SEQUENCE_NAME)
SEQUENCE_OUTPUT_PATH = os.path.join(SEQUENCE_PATH, "results")

IMAGES_PATH = SEQUENCE_PATH

################## SETUP PARAMETERS ##################

# Use pre-rectified images, seeks for "rect_" + IMAGES_PATH
USE_PRE_RECT_IMAGES = False # Don't use pre-rectified images, as COLMAP expects original images

USE_INITIAL_POSES = None # "None or camera_poses_icp.pkl" or "camera_poses_tags.pkl"

# Min an max limits of object with respect his own coordinate system (for cropping triangulated point cloud)
MAX_OBJECT_SIZE = 35 # max object base side size in mm
MAX_OBJECT_HEIGHT = 200 # max object height in mm

################## SETTING ENVIRONMENT AND LOADING CALIBRATION ##################

if USE_INITIAL_POSES:
    init_poses_file = os.path.join(SEQUENCE_OUTPUT_PATH, USE_INITIAL_POSES)

# input images
if USE_PRE_RECT_IMAGES:
    rectified_path = os.path.join(DATASETS_PATH, "captures", "rect_" + SEQUENCE_NAME)
    input_dir = rectified_path
else:
    input_dir = IMAGES_PATH

# Known object to detect in images
tag_family = "tag25h9"

# read calibration files
calibration = read_pickle(CALIB_STEREO_FILE)
maps = read_pickle(UNDISTORT_MAPS_FILE)

# separate calibration params
left_K = calibration["left_K"]
left_dist = calibration["left_dist"]
right_K = calibration["right_K"]
right_dist = calibration["right_dist"]
image_size = calibration["image_size"]
T = calibration["T"]

left_map_x = maps["left_map_x"]
left_map_y = maps["left_map_y"]
right_map_x = maps["right_map_x"]
right_map_y = maps["right_map_y"]
P1 = maps["P1"] # rectified projection matrix for left camera
P2 = maps["P2"] # rectified projection matrix for right camera
Q = maps["Q"]

# baseline in mm as x-axis distance from left to right camera
baseline_mm = T[0]

# Load ICP corrected poses
if USE_ICP_CORRECTED_POSES:
    icp_o_T_c = read_pickle(icp_poses_file)

left_file_names, right_file_names = prepare_imgs(input_dir)

################ RECONSTRUCTION and VISUALIZATION ################

# Object point cloud
object_cloud = o3d.geometry.PointCloud()

# Tags positions and colors: corner 0 - red, corner 1 - green, corner 2 - blue, corner 3 - magenta
tags_cloud = o3d.geometry.PointCloud()
corner_colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,1]], dtype = np.float64)

# Create scene
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add coordinate axis
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25.0, origin=[0, 0, 0])
vis.add_geometry(axis)

vis.add_geometry(object_cloud)
vis.add_geometry(tags_cloud)

view_ctr = vis.get_view_control()

# This should be called after adding big geometries
# as Open3D needs to resets the rendering bounding box (which defaults the camera view)
def reset_view(view_ctr):
    view_ctr.set_front([1, -1, 1])
    view_ctr.set_lookat([0, 0, 0])
    view_ctr.set_up([0, 0, 1])
    view_ctr.set_zoom(1.5)

    vis.poll_events()
    vis.update_renderer()

reset_view(view_ctr)

print("############# SfM: COLMAP #############")

output_path = pathlib.Path(SEQUENCE_OUTPUT_PATH) / "colmap"
image_dir = pathlib.Path(IMAGES_PATH)

output_path.mkdir()
mvs_path = output_path / "mvs"
database_path = output_path / "database.db"

pycolmap.extract_features(database_path, image_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
# dense reconstruction
pycolmap.undistort_images(mvs_path, output_path, image_dir)
pycolmap.patch_match_stereo(mvs_path)  # requires compilation (from source) with CUDA
pycolmap.stereo_fusion(mvs_path / "object_point_cloud_sfm.ply", mvs_path)

# Load the point cloud
object_cloud = o3d.io.read_point_cloud(mvs_path / "object_point_cloud_sfm.ply")

o3d.visualization.draw_geometries([object_cloud])

##################### VISUALIZATION #####################

vis.run()

# Saving results
# TODO: SAVE POSES!
