import os

import numpy as np
import cv2

from termcolor import colored
import pickle

from disparity.methods import StereoMethod
from utils.utils import read_pickle
from utils.images import prepare_imgs, process_images

################## DATASET SELECTION ##################
DATASETS_PATH = "data/stereo"
SEQUENCE_NAME = "raiz_apriltags_camaramovil"

CALIB_STEREO_FILE = os.path.join(DATASETS_PATH, "stereo_calibration.pkl")
UNDISTORT_MAPS_FILE = os.path.join(DATASETS_PATH, "stereo_maps.pkl")

SEQUENCE_PATH = os.path.join(DATASETS_PATH, "captures", SEQUENCE_NAME)

IMAGES_PATH = SEQUENCE_PATH
# Create a different rectified sequence
RECTIFIED_IMAGES_OUTPUT_PATH = os.path.join(DATASETS_PATH, "captures", "rect_" + SEQUENCE_NAME)

################## SETUP PARAMETERS ##################



################## SETTING ENVIRONMENT AND LOADING CALIBRATION ##################

# input images
input_dir = IMAGES_PATH

if not os.path.exists(RECTIFIED_IMAGES_OUTPUT_PATH):
    os.makedirs(RECTIFIED_IMAGES_OUTPUT_PATH)

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

left_file_names, right_file_names = prepare_imgs(input_dir)

################ IMAGE PROCESSING ################

for i, (left_file_path, right_file_path) in enumerate(zip(left_file_names, right_file_names)):
        
        print(f"Processing {left_file_path} and {right_file_path}")

        image_size, left_color, right_color = process_images(left_file_path, right_file_path, image_size)
        left_image = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
        right_image = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

        left_size = (left_image.shape[1], left_image.shape[0])
        right_size = (right_image.shape[1], right_image.shape[0])

        # rectify images
        left_image_rectified = cv2.remap(left_image, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_image_rectified = cv2.remap(right_image, right_map_x, right_map_y, cv2.INTER_LINEAR)

        # save rectified images
        left_dir, left_file_name = os.path.split(left_file_path)
        right_dir, right_file_name = os.path.split(right_file_path)

        left_rectified_file = os.path.join(RECTIFIED_IMAGES_OUTPUT_PATH, f"rect_{left_file_name}")
        right_rectified_file = os.path.join(RECTIFIED_IMAGES_OUTPUT_PATH, f"rect_{right_file_name}")

        print(f"Rectified {left_rectified_file} and {right_rectified_file}")
        cv2.imwrite(left_rectified_file, left_image_rectified)
        cv2.imwrite(right_rectified_file, right_image_rectified)