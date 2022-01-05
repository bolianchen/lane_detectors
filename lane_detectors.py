import numpy as np
import cv2
from image_processing import img_reader, gen_color_mask, rgb2gray, \
                             find_lanes, display_lines 

def lane_detector_a(img, polygon_selector, lane_colors=[]):
    """ Detect car lanes on a input img
    This detector is made by revising codes from:
    https://medium.com/analytics-vidhya/building-a-lane-detection\
    -system-f7a727c6694, which is a canny detection and Hough
    transform based lane detector.

    Implemented improvements:
        1. change the originally hard-coded triangular region to be
           a polygon determined by the user by simple interactive GUIs
        2. revise the mechanism to estimate the left and right lane,
           the cited website naively averages the lines found from the
           Hough Line Transform.

    args:
        img: a numpy array or the path to an image
        polygon_selector: a function to let users select a polygon
        lane_colors: possible lane colors
    """

    if not isinstance(img, np.ndarray):
        img = img_reader(img)

    img = np.copy(img)
    gray = rgb2gray(img)

    # selection of the Gaussian kernel size would affect
    # the performance
    gaus_blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gaus_blur, 50, 150)
    polygon = polygon_selector(edges)
    if lane_colors:
        color_mask = gen_color_mask(img, colors=lane_colors)
        polygon = cv2.bitwise_and(polygon, color_mask)
    lines = cv2.HoughLinesP(
            polygon, 2, np.pi/180, 100,
            np.array([]), minLineLength=40, maxLineGap=20)
    if lines is None:
        print('no car lanes detected')
        return img
    lane_endpts = find_lanes(img, lines)
    lanes_in_red = display_lines(img, lane_endpts)
    lanes = cv2.addWeighted(img, 0.8, lanes_in_red, 1, 1)
    return lanes
