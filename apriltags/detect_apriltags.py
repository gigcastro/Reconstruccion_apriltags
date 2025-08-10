import numpy as np
import cv2
import glob
import apriltag

from base_tags_definition import get_tag_positions

def detect_apriltags(image, tag_family):
    # Detector de AprilTags
    options = apriltag.DetectorOptions(families=tag_family, quad_decimate=0, refine_edges=True, refine_decode=True, refine_pose=True)  # importante aclarar la familia de tags!!!
    detector = apriltag.Detector(options=options)

    detections = detector.detect(image)

    object_points = []
    image_points = []
    tag_positions_mm = get_tag_positions()

    # Filter detections with decision_margin >= 30, but ensure at least one detection is kept
    filtered_detections = [d for d in detections if d.decision_margin >= 30]
    if not filtered_detections and len(detections) > 0:
        filtered_detections = [max(detections, key=lambda d: d.decision_margin)]
    detections = filtered_detections

    for detection in detections:
        tag_id = detection.tag_id
        print(f"Detected tag {tag_id}, hamming: {detection.hamming}, goodness: {detection.goodness}, decision_margin: {detection.decision_margin}")

        object_corners = tag_positions_mm[tag_id]
        image_corners = detection.corners

        object_points.append(object_corners)
        image_points.append(image_corners)

    object_points = np.concatenate(object_points)
    image_points = np.concatenate(image_points)

    return object_points, image_points