import os
import re

import pathlib
import sqlite3

import numpy as np
import pickle
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as SciRot

from utils.utils import read_pickle
import utils.colmap_utils as colmap_utils
from utils.images import prepare_imgs
import pycolmap

################## DATASET SELECTION ##################
DATASETS_PATH = "datasets/stereo_raices"
SEQUENCE_NAME = "raiz_apriltags_camaramovil" # COLMAP expects non rectified images

CALIB_STEREO_FILE = os.path.join(DATASETS_PATH, "stereo_calibration.pkl")
UNDISTORT_MAPS_FILE = os.path.join(DATASETS_PATH, "stereo_maps.pkl")

SEQUENCE_PATH = os.path.join(DATASETS_PATH, "captures", SEQUENCE_NAME)
SEQUENCE_OUTPUT_PATH = os.path.join(SEQUENCE_PATH, "results")

IMAGES_PATH = SEQUENCE_PATH

################## SETUP PARAMETERS ##################

# NOTE: Pose priors are important as they inject scale information into the reconstruction.
#       Along with the stereo baseline, they help to get a reconstruction with correct scale.

# USE_POSE_PRIORS = None # "None or camera_poses_icp.pkl" or "camera_poses_tags.pkl"
USE_POSE_PRIORS = os.path.join(DATASETS_PATH, "captures", "rect_raiz_apriltags_camaramovil", "results", "camera_poses_tags.pkl")

# Min an max limits of object with respect his own coordinate system (for cropping triangulated point cloud)
MAX_OBJECT_SIZE = 35 # max object base side size in mm
MAX_OBJECT_HEIGHT = 200 # max object height in mm

DENSE_RECONSTRUCTION = False # if True, COLMAP will perform dense reconstruction (REQUIRES CUDA COMPILATION)

################## SETTING ENVIRONMENT AND LOADING CALIBRATION ##################

if USE_POSE_PRIORS:
    init_poses_file = USE_POSE_PRIORS

# input images
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

# Transformation from left camera to right camera
cr_T_cl = np.eye(4)
cr_T_cl[0:3, 0:3] = calibration["R"]
cr_T_cl[0:3, 3] = np.reshape(calibration["T"], (3,))
cl_T_cr = np.linalg.inv(cr_T_cl)

print(pycolmap.Sim3d(cl_T_cr[0:3, 0:4]))

left_file_names, right_file_names = prepare_imgs(input_dir)

# Load ICP corrected poses
if USE_POSE_PRIORS:
    # Load camera extrinsics (list of 4x4 matrices)
    with open(init_poses_file, "rb") as f:
        init_o_T_c = pickle.load(f)

    left_pose_priors = []
    right_pose_priors = []

    for left_o_T_c in init_o_T_c:
        left_pose_priors.append(left_o_T_c)

        # Compute right camera position from left camera position
        right_o_T_c = np.dot(left_o_T_c, cl_T_cr)
        right_pose_priors.append(right_o_T_c)

################ SETTING VISUALIZATION ################

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
    vis.get_render_option().background_color = [128, 128, 128]
    vis.update_renderer()
    vis.get_render_option().background_color = [128, 128, 128]

reset_view(view_ctr)

print("############# SfM: COLMAP #############")

colmap_ws_path = pathlib.Path(SEQUENCE_OUTPUT_PATH) / "colmap"
image_path = pathlib.Path(IMAGES_PATH)

print(f"COLMAP Workspace: {colmap_ws_path}")

# Allow re-running without errors
colmap_ws_path.mkdir(parents=True, exist_ok=True)
mvs_path = colmap_ws_path / "mvs"
database_path = colmap_ws_path / "database.db"

print("############# SfM: EXTRACTING AND MATCHING FEATURES #############")
pycolmap.extract_features(database_path, image_path)
pycolmap.match_exhaustive(database_path)

if USE_POSE_PRIORS:
    colmap_utils.write_position_priors_to_database(database_path, left_file_names, left_pose_priors,
                                                   right_file_names, right_pose_priors,
                                                   std=1.0, coordinate_system=1)

print("############# SfM: 3D SPARSE RECONSTRUCTION #############")
pipeline_options = pycolmap.IncrementalPipelineOptions()
if USE_POSE_PRIORS:
    pipeline_options.use_prior_position = True

reconstructions = pycolmap.incremental_mapping(database_path, image_path, colmap_ws_path, pipeline_options)
reconstruction = reconstructions[max(reconstructions.keys())] # Sometimes pycolmap returns multiple reconstructions, we take the last one

print("############# SfM: TRANSFORM AND CROPPING SPARSE RECONSTRUCTION #############")

# Transform the whole reconstruction to align the first camera with the first tag detection
if USE_POSE_PRIORS:

    # COLMAP reconstruction can deliberately discard images if they are not useful.
    # We need to select one of them and using it as anchor to transform the reconstruction with respect to the tag detections.
    #
    # Pose priors (left_position_priors) are in a matching order as the number in the file names 

    # Find all image names matching the pattern "left_*.jpg"
    left_image_names = []
    pattern = re.compile(r"left_(\d+)\.jpg")
    for image in reconstruction.images.values():
        match = pattern.match(image.name)
        if match:
            idx = int(match.group(1))
            left_image_names.append((idx, image.name))
    # Get the image name with the lowest index
    if left_image_names:
        left_image_names.sort()
    else:
        raise ValueError("No left_*.jpg images found in reconstruction.")
    anchor_left_image = left_image_names[0][1]
    idx_left_image = left_image_names[0][0]
    print(f"Anchor image: {anchor_left_image} (index {idx_left_image})")
   
    # Get the first image in the reconstruction
    anchor_image = reconstruction.find_image_with_name(anchor_left_image)

    # Selecting an anchor pose camera from the reconstruction
    anchor_c_T_w = np.vstack((anchor_image.cam_from_world().matrix(), [0, 0, 0, 1]))
    prior_w_T_c = left_pose_priors[idx_left_image]

    # Relative transformation between the anchor camera and where the tag detecting says that is supposed to be
    new_T_old = prior_w_T_c @ anchor_c_T_w
    new_T_old = pycolmap.Sim3d(new_T_old[0:3, 0:4])

    # Transform the whole reconstruction
    reconstruction.transform(new_T_old)

# Cropping the point cloud to the object area
bbox = pycolmap.AlignedBox3d([-MAX_OBJECT_SIZE, -MAX_OBJECT_SIZE, -MAX_OBJECT_HEIGHT], [MAX_OBJECT_SIZE, MAX_OBJECT_SIZE, MAX_OBJECT_HEIGHT])
reconstruction = reconstruction.crop(bbox)

reconstruction.write(colmap_ws_path)
print(reconstruction.summary())

################ VISUALIZATION ################

for image_id, image in reconstruction.images.items():
    if not (image.has_camera_id() and image.has_pose):
        print(f"Image {image.image_id} does not have a valid pose or camera ID, skipping.")
        continue

    camera = image.camera
    pose = image.cam_from_world()
    c_T_w = np.vstack((pose.matrix(), [0, 0, 0, 1]))

    camera_frustum = o3d.geometry.LineSet.create_camera_visualization(view_width_px=camera.width,
                                                                    view_height_px=camera.height,
                                                                    intrinsic=camera.calibration_matrix(),
                                                                    extrinsic=c_T_w)
    #camera_frustum.scale(10, camera_frustum.get_center())
    vis.add_geometry(camera_frustum)
reset_view(view_ctr)

sparse_point_cloud = o3d.geometry.PointCloud()
for point3D_id, point3D in reconstruction.points3D.items():
    sparse_point_cloud.points.append(point3D.xyz)
    #sparse_point_cloud.colors.append(point3D.color)
    sparse_point_cloud.colors.append(np.array([0,0,0])) # black for better visibility
vis.add_geometry(sparse_point_cloud)

# Dense reconstruction
if DENSE_RECONSTRUCTION:
    pycolmap.undistort_images(mvs_path, colmap_ws_path, image_path)
    pycolmap.patch_match_stereo(mvs_path)  # requires compilation (from source) with CUDA

    fusion_options = pycolmap.StereoFusionOptions()
    fusion_options.bounding_box = (np.array([-MAX_OBJECT_SIZE, -MAX_OBJECT_SIZE, -MAX_OBJECT_HEIGHT], np.dtype(np.float32)), np.array([MAX_OBJECT_SIZE, MAX_OBJECT_SIZE, MAX_OBJECT_HEIGHT], np.dtype(np.float32)))
    pycolmap.stereo_fusion(mvs_path / "object_point_cloud_sfm.ply", mvs_path)

    # Load the point cloud
    object_cloud = o3d.io.read_point_cloud(mvs_path / "object_point_cloud_sfm.ply")
    vis.add_geometry(object_cloud)

##################### VISUALIZATION #####################

vis.run()
