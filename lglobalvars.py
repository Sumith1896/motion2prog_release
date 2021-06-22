import numpy as np
from scipy.optimize import curve_fit, minimize, least_squares
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

ARGS = {}
synt_args = {}

## Colors for quick use
# (B, G, R) format for OpenCV
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (192, 192, 192)
DGREY = (70, 70, 70)
ORANGE = (20, 72, 221)
GREEN = (143, 223, 4)
BLUE = (255, 144, 30)
DPINK = (84, 4, 223)
# (R, G, B) format colors
R_ORANGE = (221, 72, 20)
R_GREEN = (4, 223, 143)
R_BLUE = (30, 144, 255)
R_DPINK = (223, 4, 84)

# MPII 15 joints
mpii_joints = list(range(15))
mpii_limbs = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 1), (6, 5), (7, 6), \
            (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14)]

# PoseWarp 14 joints
posewarp_joints = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]
posewarp_limbs = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 1), (6, 5), (7, 6), \
            (1, 9), (9, 10), (10, 11), (1, 12), (12, 13), (13, 14)]

# COCO 18 joints
coco_joints = list(range(18))
coco_limbs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], \
		   [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], \
		   [0, 15], [15, 17]] 

H = None
W = None
filename = None
center = None
scale = None
scale_factor = 1.14
