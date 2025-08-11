import open3d as o3d
import os
import pickle
import numpy as np

from utils.utils import read_pickle

DATASETS_PATH = "datasets/stereo_raices"
SEQUENCE_NAME = "rect_raiz_apriltags_camaramovil"

SEQUENCE_PATH = os.path.join(DATASETS_PATH, "captures", SEQUENCE_NAME)
SEQUENCE_OUTPUT_PATH = os.path.join(SEQUENCE_PATH, "results")

CALIB_STEREO_FILE = os.path.join(DATASETS_PATH, "stereo_calibration.pkl")
UNDISTORT_MAPS_FILE = os.path.join(DATASETS_PATH, "stereo_maps.pkl")

POSES_FILE = os.path.join(SEQUENCE_OUTPUT_PATH, "camera_poses_icp.pkl")
POINT_CLOUD_FILE = os.path.join(SEQUENCE_OUTPUT_PATH, "object_point_cloud_icp.ply")
# POSES_FILE = os.path.join(SEQUENCE_OUTPUT_PATH, "camera_poses_tags.pkl")
# POINT_CLOUD_FILE = os.path.join(SEQUENCE_OUTPUT_PATH, "object_point_cloud_tags.ply")

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

# Load camera extrinsics (list of 4x4 matrices)
with open(POSES_FILE, "rb") as f:
    camera_poses = pickle.load(f)

# Load point cloud
point_cloud = o3d.io.read_point_cloud(POINT_CLOUD_FILE)

axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25.0, origin=[0, 0, 0])

intrinsic = left_K[:3, :3]
view_width_px = image_size[0]
view_height_px = image_size[1]

# Create camera frustums
camera_frustums = []
for o_T_c in camera_poses:
    c_T_o = np.linalg.inv(o_T_c)

    frustum = o3d.geometry.LineSet.create_camera_visualization(
        view_width_px=view_width_px,
        view_height_px=view_height_px,
        intrinsic=intrinsic,
        extrinsic=c_T_o
    )
    camera_frustums.append(frustum)

# Visualize point cloud and camera frustums
o3d.visualization.draw_geometries(
    [axis, point_cloud, *camera_frustums],
    window_name="Point Cloud and Camera Frustums"
)