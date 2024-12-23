import numpy as np
import cv2
import open3d as o3d

from positions import get_tag_positions
from calibration.stereo_calibration import do_calibration
from calibration.undistort import undistort
from calibration.calib import detect_apriltags
from disparity.disp import get_disparity_method, compute_disparity
from utils import read_pickle
from images import prepare_imgs, process_images
from filter_point_cloud import filter_point_cloud

################## SETUP

calib_images_dir = "data/stereo/calib/24mm_board"
calib_results_dir = "data/stereo"
checkerboard = (9,6)
square_size_mm = 24.2

captures_dir = "data/stereo/captures/raiz_apriltags_ordenadas"
rectified_dir = "data/stereo/captures/rect_raiz_apriltags_ordenadas"

################# CALIBRATION

# calculate calibration
calib_results, maps = do_calibration(
    checkerboard=checkerboard,
    square_size=square_size_mm,
    calib_images_dir=calib_images_dir,
    calib_results_dir=calib_results_dir
)

# create undistortion maps
undistort(
    calib_results,
    maps,
    captures_dir,
    rectified_dir
)

################ RECONSTRUCTION

# input images
input_dir = "data/stereo/captures/raiz_apriltags_ordenadas"

# Known object to detect in images
tag_family = "tag25h9"

# read calibration files
calib_file = "data/stereo/stereo_calibration.pkl"
maps_file = "data/stereo/stereo_maps.pkl"
calibration = read_pickle(calib_file)
maps = read_pickle(maps_file)

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
P1 = maps["P1"]
P2 = maps["P2"]
Q = maps["Q"]

baseline_mm = np.linalg.norm(T)

# configures model, defines disparity method and returns calibration object
method = get_disparity_method(
        image_size,
        P1,
        baseline_meters = baseline_mm / 1000
    )

left_file_names, right_file_names = prepare_imgs(input_dir)

all_points_3d = np.empty((0, 3))
all_colors = np.empty((0, 3))
all_camera_extrinsics = []
export_num = 0


for left_file_name, right_file_name in zip(
        left_file_names, right_file_names):

        image_size, left_image, right_image = process_images(left_file_name, right_file_name, image_size)
        left_size = (left_image.shape[1], left_image.shape[0])
        right_size = (right_image.shape[1], right_image.shape[0])

        # rectify images
        left_image_rectified = cv2.remap(left_image, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_image_rectified = cv2.remap(right_image, right_map_x, right_map_y, cv2.INTER_LINEAR)

        ######### POSE ##################

        left_object_points, left_image_points = detect_apriltags(left_image_rectified,tag_family)
        right_object_points, right_image_points = detect_apriltags(right_image_rectified, tag_family)

        left_color = cv2.cvtColor(left_image_rectified, cv2.COLOR_GRAY2BGR)
        right_color = cv2.cvtColor(right_image_rectified, cv2.COLOR_GRAY2BGR)

        # Verificar si se encontraron suficientes puntos
        if len(left_image_points) < 1 or len(right_image_points) < 1:
            print("No se encontraron suficientes AprilTags en ambas imágenes.")
            continue

        # rvec rota puntos del sistema de coordenadas del objeto al sistema de coordenadas de la cámara
        # rvec is the rotation vector and tvec is the movement vector of the cameras in respect to the world origin
        ret, rvec, tvec = cv2.solvePnP(
            left_object_points,
            left_image_points,
            left_K,
            left_dist,
            flags=cv2.SOLVEPNP_EPNP
        )

        # Armamos la matriz de transformación homogénea que convierte puntos del sistema de coordenadas del objeto a la cámara y vice versa
        c_R_o = cv2.Rodrigues(rvec)
        c_T_o = np.column_stack((c_R_o[0], tvec))
        c_T_o = np.vstack((c_T_o, [0, 0, 0, 1])) # T 4x4 que transforma puntos c_x = c_T_o  * o_x (en coordenadas del objeto a coodenadas de la cámara)
        o_T_c = np.linalg.inv(c_T_o) # T 4x4 que transforma puntos o_x = o_T_c  * c_x (en coordenadas de la camara a coodenadas del objeto)
        # o_T_c = np.column_stack((c_R_o[0], tvec))
        # o_T_c = np.vstack((o_T_c, [0, 0, 0, 1]))
        # c_T_o = np.linalg.inv(o_T_c)

        print(o_T_c)

        ############ DISPARITY #######################
        # disparity map
        disparity = compute_disparity(
            method,
            left_image_rectified,
            right_image_rectified
        )

        ############# TRIANGULATION #############

        points_3d = cv2.reprojectImageTo3D(disparity, Q)

        point_cloud = points_3d.reshape(-1, points_3d.shape[-1])
        good_points = ~np.isinf(point_cloud).any(axis=1)

        # image texture
        colors = cv2.cvtColor(left_color, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        colors = colors.reshape(-1, points_3d.shape[-1])

        # color point clouds
        # unique_color = np.random.rand(1, 3)  # Genera un color RGB aleatorio
        # colors = np.tile(unique_color, (point_cloud.shape[0], 1))  # Asignar el color a todos los puntos

        point_cloud = point_cloud[good_points]
        colors = colors[good_points]

        # Transforming points to take them to the object based coordinate system
        point_cloud = o_T_c @ np.vstack((point_cloud.T, np.ones(point_cloud.shape[0])))
        point_cloud = point_cloud[:3].T

        # Filter points so only the parts of interest of the scene are reconstructed
        point_cloud, colors = filter_point_cloud(point_cloud, colors,"raiz_apriltags")

        all_points_3d = np.vstack((all_points_3d, point_cloud))
        all_colors = np.vstack((all_colors, colors))
        all_camera_extrinsics.append(c_T_o)

##################### VISUALIZATION

# Creating scene coordinate axis (should be on pattern's corner)
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=[0, 0, 0])

# Creating pointcloud object with calculated points and colors
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_points_3d)
pcd.colors = o3d.utility.Vector3dVector(all_colors)

print(left_K.dtype)
print(o_T_c.dtype)

# Visualizing cameras
camera_frustums = []
for c_T_o in all_camera_extrinsics:
    camera_frustum = o3d.geometry.LineSet.create_camera_visualization(view_width_px=left_size[1], view_height_px=left_size[0], intrinsic=left_K[:3, :3],
                                                                        extrinsic=c_T_o)
    camera_frustum.scale(10, camera_frustum.get_center())
    camera_frustums.append(camera_frustum)

# Final scene rendering
o3d.visualization.draw_geometries([pcd]) #axis, *camera_frustums

# Saving point cloud
output_file = "results/nube_de_puntos.ply"
o3d.io.write_point_cloud(output_file, pcd)
