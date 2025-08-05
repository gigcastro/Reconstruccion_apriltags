import numpy as np

def filter_point_cloud(point_cloud, colors, filter='budha'):
    if filter == 'budha':
        point_cloud = mascara_budha(point_cloud)
    if filter == 'raiz_apriltags':
        point_cloud, colors = mascara_apriltags(point_cloud, colors)
    return point_cloud, colors

def mascara_budha(point_cloud):
    x_min, x_max = 0, 10000
    y_min, y_max = -500, -100
    z_min, z_max = -10000, 10

    mask = (
            (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
            (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) &
            (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)
    )

    # filtering points according to the mask
    point_cloud = point_cloud[mask]
    #colors = colors[mask]

    return point_cloud

def mascara_apriltags(point_cloud, colors):
    x_min, x_max = -50, 50
    y_min, y_max = -50, 50
    z_min, z_max = -200, 200

    mask = (
            (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max) &
            (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max) &
            (point_cloud[:, 2] >= z_min) & (point_cloud[:, 2] <= z_max)
    )

    # filtering points according to the mask
    point_cloud = point_cloud[mask]
    colors = colors[mask]

    return point_cloud, colors