import os

import numpy as np
import cv2
import open3d as o3d

from termcolor import colored
import pickle

from apriltags.detect_apriltags import detect_apriltags
from disparity.disp import get_disparity_method, compute_disparity
from disparity.methods import StereoMethod
from utils.utils import read_pickle
from utils.images import prepare_imgs, process_images

def np_to_o3d_pointcloud(point_cloud, colors):
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    cloud_o3d.colors = o3d.utility.Vector3dVector(colors)
    return cloud_o3d

################## DATASET SELECTION ##################
DATASETS_PATH = "datasets/stereo_raices"
SEQUENCE_NAME = "rect_raiz_apriltags_camaramovil"

CALIB_STEREO_FILE = os.path.join(DATASETS_PATH, "stereo_calibration.pkl")
UNDISTORT_MAPS_FILE = os.path.join(DATASETS_PATH, "stereo_maps.pkl")

SEQUENCE_PATH = os.path.join(DATASETS_PATH, "captures", SEQUENCE_NAME)
SEQUENCE_OUTPUT_PATH = os.path.join(SEQUENCE_PATH, "results")

IMAGES_PATH = SEQUENCE_PATH

################## SETUP PARAMETERS ##################

# Set to False when using a "rect_" sequence
RECTIFY_IMAGES = False

# Min an max limits of object with respect his own coordinate system (for cropping triangulated point cloud)
MAX_OBJECT_SIZE = 35 # max object base side size in mm
MAX_OBJECT_HEIGHT = 200 # max object height in mm

# DISPARITY METHOD
DISPARITY_METHOD_NAME = "OpenCV_SGBM"  # or "OpenCV_BM", "OpenCV_SGBM", "CREStereo"

################## SETTING ENVIRONMENT AND LOADING CALIBRATION ##################

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

# baseline in mm as x-axis distance from left to right camera
baseline_mm = T[0]

# configures model, defines disparity method and returns calibration object
stereo_matcher = get_disparity_method(
        image_size,
        P1[0,0], P1[1,1], P1[0,2], P2[0,2], P1[1,2],
        baseline_meters = baseline_mm / 1000,
        method_name = DISPARITY_METHOD_NAME
    )

left_file_names, right_file_names = prepare_imgs(input_dir)

################ RECONSTRUCTION and VISUALIZATION ################

# Camera poses and tags positions 
cameras_extrinsics = []

# Accumulated object point cloud
object_cloud = o3d.geometry.PointCloud()
# Tags positions and colors: corner 0 - red, corner 1 - green, corner 2 - blue, corner 3 - magenta
tags_cloud = o3d.geometry.PointCloud()
corner_colors = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,1]], dtype = np.float64)

# Create scene
vis = o3d.visualization.VisualizerWithKeyCallback()
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

# Register key callback to stop loop
stopped = False
def on_key_pressed(vis):
    global stopped
    print("Key 'S' pressed. Stopping loop.")
    stopped = True
vis.register_key_callback(ord('S'), on_key_pressed)

################ IMAGE PROCESSING ################

for i, (left_file_name, right_file_name) in enumerate(zip(left_file_names, right_file_names)):
        
        if stopped:
            print("Processing loop stopped.")
            break
        
        print(f"Processing {left_file_name} and {right_file_name}")

        image_size, left_color, right_color = process_images(left_file_name, right_file_name, image_size)
        left_image = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

        left_size = (left_image.shape[1], left_image.shape[0])
        right_size = (right_image.shape[1], right_image.shape[0])

        # Rectify images if required
        if RECTIFY_IMAGES:
            left_image_rectified = cv2.remap(left_image, left_map_x, left_map_y, cv2.INTER_LINEAR)
            right_image_rectified = cv2.remap(right_image, right_map_x, right_map_y, cv2.INTER_LINEAR)
        else:
            left_image_rectified = left_image
            right_image_rectified = right_image

        ######### POSE ##################

        left_object_points, left_image_points = detect_apriltags(left_image_rectified,tag_family)
        #right_object_points, right_image_points = detect_apriltags(right_image_rectified, tag_family)

        # Verificar si se encontraron suficientes puntos
        if len(left_image_points) < 1:
            print("No se encontraron suficientes AprilTags en la imagen izquierda.")
            continue

        if left_object_points.size == 4:
            pnp_method = cv2.SOLVEPNP_IPPE
        else:
            pnp_method = cv2.SOLVEPNP_EPNP
        
        # rvec rota puntos del sistema de coordenadas del objeto al sistema de coordenadas de la cámara
        # rvec is the rotation vector and tvec is the movement vector of the cameras in respect to the world origin
        ret, rvec, tvec = cv2.solvePnP(
            left_object_points,
            left_image_points,
            left_K,
            left_dist,
            flags=pnp_method
        )

        # Armamos la matriz de transformación homogénea que convierte puntos del sistema de coordenadas del objeto a la cámara y vice versa
        c_R_o = cv2.Rodrigues(rvec)
        c_T_o = np.column_stack((c_R_o[0], tvec))
        c_T_o = np.vstack((c_T_o, [0, 0, 0, 1])) # T 4x4 que transforma puntos c_x = c_T_o  * o_x (en coordenadas del objeto a coodenadas de la cámara)
        o_T_c = np.linalg.inv(c_T_o) # T 4x4 que transforma puntos o_x = o_T_c  * c_x (en coordenadas de la camara a coodenadas del objeto)
        # o_T_c = np.column_stack((c_R_o[0], tvec))
        # o_T_c = np.vstack((o_T_c, [0, 0, 0, 1]))
        # c_T_o = np.linalg.inv(o_T_c)

        cameras_extrinsics.append(o_T_c)

        tags_cloud.points.extend(o3d.utility.Vector3dVector(left_object_points))
        num_tags = len(left_object_points) // 4
        repeated_corner_colors = np.tile(corner_colors, (num_tags, 1))
        
        tags_cloud.colors.extend(o3d.utility.Vector3dVector(repeated_corner_colors))

        # Debug tags order
        #print("left_object_points", left_object_points)
        #print("reated_corner_colors", repeated_corner_colors)

        print("############ DISPARITY #######################")
        
        disparity = compute_disparity(
            stereo_matcher,
            left_image_rectified,
            right_image_rectified
        )

        print("############# TRIANGULATION #############")

        points_3d = cv2.reprojectImageTo3D(disparity, Q)

        # Reshape points_3d to a 2D array of shape (N, 3)
        point_cloud = points_3d.reshape(-1, points_3d.shape[-1])

        # Transforming points to take them to the object based coordinate system
        # point_cloud = o_T_c @ np.vstack((point_cloud.T, np.ones(point_cloud.shape[0])))
        # point_cloud = point_cloud[:3].T

        print("############# BUILD POINTCLOUD #############")

        # Image texture
        colors = cv2.cvtColor(left_color, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        colors = colors.reshape(-1, points_3d.shape[-1])

        # Remove points with infinite values
        good_points = ~np.isinf(point_cloud).any(axis=1)
        point_cloud = point_cloud[good_points]
        colors = colors[good_points]

        # point_cloud as open3d point cloud
        point_cloud = np_to_o3d_pointcloud(point_cloud, colors)

        # o_T_c * c_p (with c_p a point in camera coordinates)
        point_cloud = point_cloud.transform(o_T_c)  # Transform from camera to object coordinates

        print("############# CROP & FILTERING #############")

        # Filter points so only the parts of interest of the scene are reconstructed
        mins = np.array([-MAX_OBJECT_SIZE, -MAX_OBJECT_SIZE, -MAX_OBJECT_HEIGHT])
        maxs = np.array([MAX_OBJECT_SIZE, MAX_OBJECT_SIZE, MAX_OBJECT_HEIGHT])
        point_cloud = point_cloud.crop(o3d.geometry.AxisAlignedBoundingBox(mins, maxs))

        if point_cloud.is_empty():
            print(colored("WARNING:", "red") + "Triangulated point cloud was so far away from expected bounding box that it is empty after cropping and filtering.")
            print(colored("WARNING:", "red") + "Camera pose extrinsic o_T_c computed using the tags may be very bad.")
            continue

        # Remove outliers (helps to reduce point trails at the edges)
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        point_cloud = point_cloud.select_by_index(ind)

        # Accumulate object cloud
        object_cloud.points.extend(point_cloud.points)
        object_cloud.colors.extend(point_cloud.colors)
        object_cloud.normals.extend(point_cloud.normals)

        ############# UPDATE VISUALIZATION #############

        vis.update_geometry(object_cloud)
        vis.update_geometry(tags_cloud)

        camera_frustum = o3d.geometry.LineSet.create_camera_visualization(view_width_px=left_size[0], view_height_px=left_size[1], intrinsic=P1[:3, :3], extrinsic=c_T_o)
        #camera_frustum.scale(10, camera_frustum.get_center())
        vis.add_geometry(camera_frustum)

        reset_view(view_ctr)

# Lock visualization thread
reset_view(view_ctr)
vis.run()

if stopped:
    print("Processing loop stopped by user, results won't be saved.")
    exit(0)

# Saving point cloud
print("Saving point cloud...")

if not os.path.exists(SEQUENCE_OUTPUT_PATH):
    os.makedirs(SEQUENCE_OUTPUT_PATH)

output_file = os.path.join(SEQUENCE_OUTPUT_PATH, "object_point_cloud_tags.ply")
o3d.io.write_point_cloud(output_file, object_cloud)
print("Saving camera poses o_T_c estimated by tags ...")
output_file = os.path.join(SEQUENCE_OUTPUT_PATH, "camera_poses_tags.pkl")
with open(output_file, "wb") as f:
    pickle.dump(cameras_extrinsics, f)