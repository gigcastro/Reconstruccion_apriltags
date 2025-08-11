from pathlib import Path

from disparity.methods import Config, Calibration, InputPair
from disparity.method_cre_stereo import CREStereo
from disparity.method_opencv_bm import StereoBM
from disparity.method_opencv_bm import StereoSGBM


def get_disparity_method(
        image_size,
        fx, fy, cx0, cx1, cy, # left f, left cx, right cx, both cy
        baseline_meters,
        method_name = "OpenCV_SGBM"
):
    models_path = Path("disparity/models")
    config = Config(models_path=models_path)

    if method_name == "OpenCV_BM":
        method = StereoBM(config)
    elif method_name == "OpenCV_SGBM":
        method = StereoSGBM(config)
    elif method_name == "CREStereo":
        method = CREStereo(config)
    else:
        raise ValueError(f"Unknown disparity method: {method_name}")

    # width an height
    w, h = image_size

    j_calib = {
        "width": w,
        "height": h,
        "baseline_meters": baseline_meters,
        "fx": fx,
        "fy": fy,
        "cx0": cx0,
        "cx1": cx1,
        "cy": cy,
        "depth_range": [0.1, 30.0],
        "left_image_rect_normalized": [0, 0, 1, 1]
    }

    # Reads json as calibration object
    calibration = Calibration(**j_calib)

    return method, calibration

def compute_disparity(
        disparity_method,
        left_image_rectified,
        right_image_rectified
):
    method, calibration = disparity_method
    pair = InputPair(left_image_rectified, right_image_rectified, calibration)
    # disparity algorithm from the disparity method defined before
    disparity = method.compute_disparity(pair)

    return disparity.disparity_pixels