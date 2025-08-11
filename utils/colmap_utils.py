import os
import pathlib
import sqlite3
from turtle import position

import numpy as np
import pickle
from scipy.spatial.transform import Rotation as SciRot

import pycolmap

# coordinate_system:int, -1: unknwon, 0: WGS84, 1: Cartesian
def write_position_priors_to_database(database_path:pathlib.Path,
                                      left_file_names:list[str], left_pose_priors:list,
                                      right_file_names:list[str], right_pose_priors:list,
                                      std:float=1.0, coordinate_system:int=-1):
    database_path = pathlib.Path(database_path)

    if not database_path.exists():
        print("ERROR: database path does not exist.")
        return

    if not left_pose_priors or len(left_pose_priors) == 0:
        print("ERROR: left position priors list is empty.")
        return

    if not right_pose_priors or len(right_pose_priors) == 0:
        print("ERROR: right position priors list is empty.")
        return

    position_covariance = np.diag([std**2,std**2,std**2])

    # Add position priors from file.
    with pycolmap.Database(database_path) as colmap_db:

        # If a pose prior is already present, writing fails, clear all pose priors first.
        colmap_db.clear_pose_priors()

        for i, (left_image_name, left_pose, right_image_name, right_pose) in enumerate(zip(left_file_names, left_pose_priors, right_file_names, right_pose_priors)):
            left_position = left_pose[0:3, 3]
            right_position = right_pose[0:3, 3]

            # save rectified images
            left_dir, left_file_name = os.path.split(left_image_name)
            right_dir, right_file_name = os.path.split(right_image_name)
            update_position_prior_from_image_name(colmap_db, left_file_name, left_position, coordinate_system, position_covariance)
            update_position_prior_from_image_name(colmap_db, right_file_name, right_position, coordinate_system, position_covariance)

def update_position_prior_from_image_name(
    colmap_db: pycolmap.Database,
    image_name: str,
    position: np.array,
    coordinate_system: int = -1,
    position_covariance: np.array = None,
):
    """
    Update the position prior for a specific image name in the database.
    If the position prior doesn't exist, insert a new one.

    Args:
        colmap_db (pycolmap.Database): colmap database to update.
        image_name (str): name of the image to update.
        position (np.array): Position as a 3-element array (x, y, z).
        coordinate_system (int): Coordinate system index (default: -1).
        position_covariance (np.array): 3x3 position covariance matrix
                                        (default: None).
    """
    # Get image_id from image_name
    if colmap_db.exists_image(image_name):
        position = np.asarray(position, dtype=np.float64).reshape(3, 1)

        if position_covariance is None:
            position_covariance = np.full((3, 3), np.nan, dtype=np.float64)

        image = colmap_db.read_image_with_name(image_name)

        colmap_db.write_pose_prior(
            image.image_id,
            pycolmap.PosePrior(
                position,
                position_covariance,
                pycolmap.PosePriorCoordinateSystem(coordinate_system),
            ),
        )
    else:
        print(f"[COLMAP Priors] Image at path {image_name} not found in database.")