import math
import numpy as np
import cv2

from matplotlib import pyplot as plt

COLOR_HSV_BOUNDARYS = {'red':   [([0,43,46], [10,255,255]), 
                                 ([156,43,46], [180,255,255])],
                       'yellow':[([20,43,46], [30,255,255])],
                       'white': [([0, 0, 100], [255, 255, 255])]}

def img_reader(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def gen_color_mask(img, colors=['red', 'yellow', 'white']):
    mask = np.zeros(img.shape[:2], dtype='uint8')
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for color in colors:
        for bd in COLOR_HSV_BOUNDARYS[color]:
            mask = cv2.bitwise_or(mask,
                                  cv2.inRange(
                                      img_hsv,
                                      np.array(bd[0], dtype='uint8'),
                                      np.array(bd[1], dtype='uint8')))
    return mask

def rgb2gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def make_polygon_selector(polygon=[]):
    """Create a function to receive a polygon selected by the user"""

    def select_polygon(img):
        """ Isolate a polygon region where the lane lines are located within 
        """

        nonlocal polygon
        if len(polygon) == 0:
            is_valid_polygon = False
            while not is_valid_polygon:
                # collect points by left-clicking on the image
                polygon = collect_clicked_coords(img)
                is_valid_polygon, mask = check_selected_polygon(img, polygon)
        else:
            assert len(polygon) >= 3
            mask = create_mask(img, polygon)
                
        return mask

    return select_polygon

def create_mask(img, polygon):
    """Return the masked img by the polygon"""
    if len(img.shape) == 2:
        fil_color = 255
    else:
        fil_color = (255, 255, 255)
    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, np.array([polygon]), fil_color)
    mask = cv2.bitwise_and(img, mask)
    return mask

def collect_clicked_coords(img):
    """Collect the point coordinates selected by mouse left-clicking
    """
    coords = []
    fig = plt.figure()
    plt.imshow(img)
    plt.title('Please use left-click to select all the vertices\n'\
              ' of a polygon enclosing the car lanes to detect.\n'\
              'Press "y" when finished')
    def onclick(event):
        ix, iy = round(event.xdata), round(event.ydata)
        nonlocal coords
        coords.append((round(ix), round(iy)))
        plt.scatter(ix, iy)
        plt.draw()
        print(f'currently selected {coords}')
    def press(event):
        pressed_key = event.key.lower()
        if pressed_key == 'y':
            plt.close(event.canvas.figure)
    cid1 = fig.canvas.mpl_connect('button_press_event', onclick)
    cid2 = fig.canvas.mpl_connect('key_press_event', press)
    plt.show()
    fig.canvas.mpl_disconnect(cid1)
    fig.canvas.mpl_disconnect(cid2)
    del fig
    return coords

def check_selected_polygon(img, coords):
    """Let the user confirm if the selected polygon is as expected"""

    is_valid_polygon = False

    if len(coords) < 3:
        print('Please select at least 3 points from the plot')
        return is_valid_polygon, None

    fig = plt.figure()
    plt.title('If the selected region looks good?\n'\
              'Press y to continue and any other key to re-select')
    def press(event):
        nonlocal is_valid_polygon
        pressed_key = event.key.lower()
        if pressed_key == 'y':
            plt.close(event.canvas.figure)
            is_valid_polygon = True
        else:
            plt.close(event.canvas.figure)
    cid = fig.canvas.mpl_connect('key_press_event', press)
    mask = create_mask(img, coords) 
    plt.imshow(mask)
    plt.show()
    fig.canvas.mpl_disconnect(cid)
    del fig

    return is_valid_polygon, mask

def find_lanes(img, lines, smallest_deg=15):
    """Determine the representative left and right lanes

    args:
        smallest_deg: the smallest included angle with horizon for a lane
    """
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.squeeze(0)
        # estimate the slope and y interception for a line
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        # radian would in (-pi/2, pi/2)
        radian = math.atan(slope)
        y_int = parameters[1]
        # ignore lines with smaller included angles with horizon
        # 15 degree is heuristically define
        if abs(radian) < (math.pi/180) * smallest_deg:
            continue
        elif slope < 0:
            left.append((slope, y_int))
        else:
            right.append((slope, y_int))

    results = []
    if right:
        right_avg = np.average(right, axis=0)
        right_line = make_points(img, right_avg)
        results.append(right_line)
    if left:
        left_avg = np.average(left, axis=0)
        left_line = make_points(img, left_avg)
        results.append(left_line)
    return np.array(results)

def make_points(img, lane_params):
    """Estimate the end points for a car lane
    """
    slope, y_int = lane_params

    # heuristics: assume a lane only appears lower than 0.6 * image height
    y1 = img.shape[0]
    y2 = int(y1 * 0.6)
    x1 = int((y1 - y_int)//slope)
    x2 = int((y2 - y_int)//slope)
    return np.array([x1, y1, x2, y2])

def display_lines(img, lines, line_color = (255,0,0)):
    """Return an black image with lines drawn
    """
    lines_img = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_img, (x1, y1), (x2, y2), line_color, 10)
    return lines_img
