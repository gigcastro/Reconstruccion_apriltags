import glob
import os
import cv2

from images import prepare_imgs, process_images


def undistort(calib_results, maps, input_dir, output_dir):

    image_size = calib_results['image_size']
    left_map_x = maps['left_map_x']
    left_map_y = maps['left_map_y']
    right_map_x = maps['right_map_x']
    right_map_y = maps['right_map_y']

    left_file_names, right_file_names = prepare_imgs(input_dir)

    for left_file_name, right_file_name in zip(
        left_file_names, right_file_names
    ):
        image_size, left_image, right_image = process_images(left_file_name, right_file_name, image_size)

        # rectify images based on calibration
        left_image_rectified = cv2.remap(left_image, left_map_x, left_map_y, cv2.INTER_LINEAR)
        right_image_rectified = cv2.remap(right_image, right_map_x, right_map_y, cv2.INTER_LINEAR)

        rleft_file_name = "rect_" + os.path.split(left_file_name)[1]
        rright_file_name = "rect_" + os.path.split(right_file_name)[1]
        output_left_file = os.path.join(output_dir, rleft_file_name)
        output_right_file = os.path.join(output_dir, rright_file_name)

        #print(f"writting undistorted images {rleft_file_name}, {rright_file_name}...")
        cv2.imwrite(output_left_file, left_image_rectified)
        cv2.imwrite(output_right_file, right_image_rectified)