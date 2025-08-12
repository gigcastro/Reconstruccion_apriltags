import os
import pickle
import pathlib

import numpy as np
import open3d as o3d
import pycolmap

from utils.utils import read_pickle

################## DATASET SELECTION ##################
DATASETS_PATH = "datasets/stereo_raices"
SEQUENCE_NAME = "raiz_apriltags_camaramovil" # COLMAP expects non rectified images

CALIB_STEREO_FILE = os.path.join(DATASETS_PATH, "stereo_calibration.pkl")
UNDISTORT_MAPS_FILE = os.path.join(DATASETS_PATH, "stereo_maps.pkl")

SEQUENCE_PATH = os.path.join(DATASETS_PATH, "captures", SEQUENCE_NAME)
SEQUENCE_OUTPUT_PATH = os.path.join(SEQUENCE_PATH, "results")

IMAGES_PATH = SEQUENCE_PATH

################## SETUP PARAMETERS ##################

# Min an max limits of object with respect his own coordinate system (for cropping triangulated point cloud)
MAX_OBJECT_SIZE = 40 # max object base side size in mm
MAX_OBJECT_HEIGHT = 200 # max object height in mm

################ SETTING VISUALIZATION ################

# Create scene
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

# Add coordinate axis
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=25.0, origin=[0, 0, 0])
vis.add_geometry(axis)

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

print(f"Reading COLMAP Workspace: {colmap_ws_path}")

mvs_path = colmap_ws_path / "mvs"
database_path = colmap_ws_path / "database.db"

# List directories in colmap_ws_path with numeric names and select the one with the greatest number
reconstruction_dirs = [int(item) for item in os.listdir(colmap_ws_path) if item.isdigit() and (colmap_ws_path / item).is_dir()]
last_reconstruction_dir = None
if not len(reconstruction_dirs) == 0:
    last_reconstruction_dir = colmap_ws_path / str(max(reconstruction_dirs)) 
    print("Visualizing the last reconstruction found in COLMAP workspace:", last_reconstruction_dir)
else:
    print("No reconstructions found in COLMAP workspace.")
    exit()

reconstruction = pycolmap.Reconstruction(last_reconstruction_dir)
print(reconstruction.summary())

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

# Key callback to toggle dense point cloud visualization
dense_pcd_path = colmap_ws_path / "object_point_cloud_sfm.ply"
dense_pcd = None
dense_pcd_visible = False

if dense_pcd_path.exists():
    print(f"Loading dense point cloud from: {dense_pcd_path}")
    print(f"Visualize the dense point cloud by pressing 'D' key.")
    dense_pcd = o3d.io.read_point_cloud(str(dense_pcd_path))
else:
    print(f"Dense point cloud file not found: {dense_pcd_path}")

#######
# TODO(gcastro): Key toggling visualization is EXTREMELY BUGGY
######

def toggle_dense_pcd(vis):
    global dense_pcd_visible, dense_pcd, sparse_point_cloud

    if not dense_pcd:
        return

    if dense_pcd_visible:
        vis.remove_geometry(dense_pcd, reset_bounding_box=False)
        vis.add_geometry(sparse_point_cloud, reset_bounding_box=False)
        dense_pcd_visible = False
        print("Dense point cloud removed.")
    else:
        vis.add_geometry(dense_pcd, reset_bounding_box=False)
        vis.remove_geometry(sparse_point_cloud, reset_bounding_box=False)
        dense_pcd_visible = True
        print("Dense point cloud added.")
    vis.update_renderer()

# Filter points so only the parts of interest of the scene are reconstructed
mins = np.array([-MAX_OBJECT_SIZE, -MAX_OBJECT_SIZE, -MAX_OBJECT_HEIGHT])
maxs = np.array([MAX_OBJECT_SIZE, MAX_OBJECT_SIZE, MAX_OBJECT_HEIGHT])
bbox = o3d.geometry.AxisAlignedBoundingBox(mins, maxs)

dense_crop = dense_pcd.crop(bbox)
sparse_crop = sparse_point_cloud.crop(bbox)

cl, ind = dense_crop.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
dense_crop = dense_crop.select_by_index(ind)

cl, ind = sparse_crop.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
sparse_crop = sparse_crop.select_by_index(ind)

cropping = False
def toggle_bbox_pcd(vis):
    global dense_pcd, sparse_point_cloud, cropping, dense_crop, sparse_crop, dense_pcd_visible

    if not cropping:        
        if dense_pcd_visible:
            vis.remove_geometry(dense_pcd, reset_bounding_box=False)
            vis.add_geometry(dense_crop, reset_bounding_box=False)
        else:
            vis.remove_geometry(sparse_point_cloud, reset_bounding_box=False)
            vis.add_geometry(sparse_crop, reset_bounding_box=False)
        cropping = True
        print("Point cloud cropped to object area.")
    else:
        if dense_pcd_visible:
            vis.remove_geometry(dense_crop, reset_bounding_box=False)
            vis.add_geometry(dense_pcd, reset_bounding_box=False)
        else:
            vis.remove_geometry(sparse_crop, reset_bounding_box=False)
            vis.add_geometry(sparse_point_cloud, reset_bounding_box=False)
        cropping = False
        print("Point cloud uncropped.")
    vis.update_renderer()

vis.register_key_callback(ord("D"), lambda vis: toggle_dense_pcd(vis))
vis.register_key_callback(ord("B"), lambda vis: toggle_bbox_pcd(vis))

vis.run()