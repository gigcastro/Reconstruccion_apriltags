import glob
import os
import cv2

from utils.utils import numeric_sort

def stereoimages_size(image_size, left_size, right_size):
    # checks that images sizes match
    if left_size != right_size:
        raise Exception(f"left and right images sizes differ: left {left_size} / right {right_size}")
    if image_size is None:
        # remembers the images size
        image_size = left_size
    else:
        if image_size != left_size:
            raise Exception(f"there are images with different sizes: {image_size} vs {left_size}")
    return image_size


def prepare_imgs(calib_images_directory):
    left_files_pattern = "*left*.jpg"
    right_files_pattern = "*right*.jpg"
    left_file_names = sorted(
        glob.glob(
            os.path.join(calib_images_directory, left_files_pattern)
        ),
        key=numeric_sort
    )
    right_file_names = sorted(
        glob.glob(
            os.path.join(calib_images_directory, right_files_pattern)
        ),
        key=numeric_sort
    )

    num_left = len(left_file_names)
    num_right = len(right_file_names)

    if num_left != num_right:
        raise Exception(f"the number of files (left {num_left} / right{num_right}) doesn't match")

    return left_file_names, right_file_names


def process_images(left_file_name,
                   right_file_name,
                   image_size):

    #print("processing", left_file_name, right_file_name)

    # read left and right images
    left_image = cv2.imread(left_file_name, cv2.IMREAD_COLOR_BGR)
    right_image = cv2.imread(right_file_name, cv2.IMREAD_COLOR_BGR)

    # get the images sizes
    left_size = (left_image.shape[1], left_image.shape[0])
    right_size = (right_image.shape[1], right_image.shape[0])

    image_size = stereoimages_size(image_size, left_size, right_size)

    return image_size, left_image, right_image

