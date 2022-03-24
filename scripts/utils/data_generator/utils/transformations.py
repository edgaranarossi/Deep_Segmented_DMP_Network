from numpy import cos,sin,deg2rad,pi
import numpy as np

def rotateCoordList(point, center, orientation):
    px, py = point
    cx, cy = center
    new_x = cos(deg2rad(orientation)) * (px-cx) - sin(deg2rad(orientation)) * (py-cy) + cx
    new_y = sin(deg2rad(orientation)) * (px-cx) + cos(deg2rad(orientation)) * (py-cy) + cy
    # print(new_x, new_y)
    return new_x, new_y

def rotateCoordNp(point, center, orientation):
    px = point[:,0]
    py = point[:,1]
    cx = center[0]
    cy = center[1]
    new_x = cos(deg2rad(orientation)) * (px-cx) - sin(deg2rad(orientation)) * (py-cy) + cx
    new_y = sin(deg2rad(orientation)) * (px-cx) + cos(deg2rad(orientation)) * (py-cy) + cy
    new_points = np.append(new_x.reshape(-1, 1), new_y.reshape(-1, 1), axis=1)
    return new_points
