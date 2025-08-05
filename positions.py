import math
import numpy as np

def get_tag_positions():
    # tag_positions_mm = {
    #     0 : [0.0, 7.0, -47.5],
    #     1 : [33.5, 7.0, -33.5],
    #     2 : [47.5, 7.0, 0.0],
    #     3 : [33.5, 7.0, 33.5],
    #     4 : [0.0, 7.0, 47.5],
    #     5 : [-33.5, 7.0, 33.5],
    #     6 : [-47.5, 7.0, 0.0],
    #     7 : [-33.5, 7.0, -33.5]
    # }

    # Tag centroids with respect to a right-handed coordinate system
    # where "the front of the object is (1,0,0)" and Z positive is up.
    tag_centroids_mm = {
        0 : [47.5, 0.0, 7.0],
        1 : [33.5, 33.5, 7.0],
        2 : [0.0, 47.5, 7.0],
        3 : [-33.5, 33.5, 7.0],
        4 : [-47.5, 0.0, 7.0],
        5 : [-33.5, -33.5, 7.0],
        6 : [0.0, -47.5, 7.0],
        7 : [33.5, -33.5, 7.0]
    }

    # Relative tag coordinates as seen the object as with X-towards_tag, Y-right, Z-up
    tag_relative_corner_positions = {
        # bottom_left, top_left, top_right, bottom_right
        0: [[0.0, -3.0, -3.0], [0.0, -3.0, 3.0], [0, 3.0, 3.0], [0, 3.0, -3.0]],
        #  top_left, top_right, bottom_right, bottom_left
        1: [[0.0, -3.0, 3.0], [0.0, 3.0, 3.0], [0.0, 3.0, -3.0], [0.0, -3.0, -3.0]],
        # bottom_left, top_left, top_right, bottom_right
        2: [[0.0, -3.0, -3.0], [0.0, -3.0, 3.0], [0.0, 3.0, 3.0], [0.0, 3.0, -3.0]],
        # top_right, bottom_right, bottom_left, top_left
        3: [[0.0, 3.0, 3.0], [0.0, 3.0, -3.0], [0.0, -3.0, -3.0], [0.0, -3.0, 3.0]],
        # top_left, top_right, bottom_right, bottom_left
        4: [[0.0, -3.0, 3.0], [0.0, 3.0, 3.0], [0.0, 3.0, -3.0], [0.0, -3.0, -3.0]],
        # top_right, bottom_right, bottom_left, top_left
        5: [[0.0, 3.0, 3.0], [0.0, 3.0, -3.0], [0.0, -3.0, -3.0], [0.0, -3.0, 3.0]],
        # bottom_right, bottom_left, top_left, top_right
        6: [[0.0, 3.0, -3.0], [0.0, -3.0, -3.0], [0.0, -3.0, 3.0], [0.0, 3.0, 3.0]],
        # top_left, top_right, bottom_right, bottom_left
        7: [[0.0, -3.0, 3.0], [0.0, 3.0, 3.0], [0.0, 3.0, -3.0], [0.0, -3.0, -3.0]]
    }

    # Compute tag_positions_mm by adding centroid and relative corners for each tag
    tag_positions_mm = {}
    for tag_id, angle in zip(tag_centroids_mm, range(0, 360, 45)):
        centroid = np.array(tag_centroids_mm[tag_id])

        # Apply relative deltas to centroids
        rel_corners = tag_relative_corner_positions[tag_id]
        abs_corners = [ (centroid + np.array(corner)).tolist() for corner in rel_corners ]
        
        # Rotate corners around the centroid with respect to the Z-axis
        abs_rotated = rotate_tag_points(abs_corners, angle)
        
        tag_positions_mm[tag_id] = abs_rotated

    #tag_positions_mm = {
        # 0: [[-3.0, 4.0, -47.5],[-3.0, 10.0, -47.5],[3.0, 10.0, -47.5],[3.0, 4.0, -47.5]],
        # 1: [[30.5, 10.0, -33.5],[36.5, 10.0, -33.5],[36.5, 4.0, -33.5],[30.5, 4.0, -33.5]],
        # 2: [[44.5, 4.0, 0.0],[44.5, 10.0, 0.0],[50.5, 10.0, 0.0],[50.0, 4.0, 0.0]],
        # 3: [[30.5, 4.0, 33.5],[30.5, 10.0, 33.5],[36.5, 10.0, 33.5],[36.5, 4.0, 33.5]],
        # 4: [[-3.0, 10.0, 47.5],[3.0, 10.0, 47.5],[3.0, 4.0, 47.5],[-3.0, 4.0, 47.5]],
        # 5: [[-36.5, 10.0, 33.5],[-30.5, 10.0, 33.5],[-30.5, 4.0, 33.5],[-36.5, 4.0, 33.5]],
        # 6: [[-50.5, 10.0, 0.0],[-44.5, 4.0, 0.0],[-44.5, 4.0, 0.0],[-50.5, 10.0, 0.0]],
        # 7: [[-36.5, 10.0, -33.5],[-30.5, 10.0, -33.5],[-30.5, 4.0, -33.5],[-36.5, 4.0, -33.5]]
    #}
    # # For each tag, store a list of rotated positions for angles 0, 45, ..., 315
    # rotated_tags_by_angle = {}
    # for (tag_id, points), angle in zip(tag_positions_mm.items(), range(0, -360, -45)):
    #     rotated = rotate_tag_points(points, angle)
    #     rotated_tags_by_angle[tag_id] = rotated

    return tag_positions_mm

# Rotate each tag's 4 points around its centroid (y-axis)
def rotate_tag_points(tag_points, degrees):
    rotated_tags = {}
    angle_rad = math.radians(degrees)

    # Rotation matrix around z-axis
    R = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad), 0],
        [math.sin(angle_rad),  math.cos(angle_rad), 0],
        [0, 0, 1]
    ])

    pts = np.array(tag_points)
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    pts_rotated = np.dot(pts_centered, R.T) + centroid
    
    return pts_rotated.tolist()

def center_to_corners(tag_center):
    top_left = [tag_center[0]-3.0, tag_center[1]+3.0, tag_center[2]]
    top_right = [tag_center[0]+3.0, tag_center[1]+3.0, tag_center[2]]
    bottom_left = [tag_center[0]-3.0, tag_center[1]-3.0, tag_center[2]]
    bottom_right = [tag_center[0]+3.0, tag_center[1]-3.0, tag_center[2]]

    corners = [top_left, top_right, bottom_left, bottom_right]
    return corners