import os
import argparse
import cv2
from matplotlib import pyplot as plt

from data_iterators import is_a_file, VideoReader, ImageReader
from image_processing import make_polygon_selector
from lane_detectors import lane_detector_a

video_exts = ['.mp4']
image_exts = ['.png', '.jpg']

DETECTORS = {'a': lane_detector_a}

# you may replace the empty list with a pre-defined list of tuples of x, y
# to form a polygon enclosing the lanes to detect
# an example for a 1920x1080 image [(747,515), (288,805), (1049,817), (882,507)]
pre_selected_polygon = []

def main(args):

    # prepare data
    input_file = args.input
    if is_a_file(input_file, video_exts):
        data = VideoReader(input_file)
    else:
        data = ImageReader(input_file, img_types=image_exts)

    # a selector to let users select the region to detect lanes
    # required by detector a
    polygon_selector = make_polygon_selector(pre_selected_polygon)
    
    for idx, img in enumerate(data):

        # video reader based on cv2 may output empty frame at the beginning
        if img is None:
            continue

        # determine the selected region interactively
        polygon_selector(img)

        # conduct lane detection
        img = DETECTORS[args.detector](img, polygon_selector)

        # draw and save images
        plt.imshow(img)
        plt.axis('off')
        plt.draw()
        if not args.not_display:
            plt.pause(0.1)
        if args.save_path:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            plt.savefig(os.path.join(args.save_path, f'{idx}.png'))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--detector',
                           type=str,
                           choices = ['a'],
                           help='which lane detector to use')
    argparser.add_argument('--input',
                           type=str,
                           help='input data in either of the 3 formats'
                                '1. path to a video file; '
                                '2. path to a image file; '
                                '3. path to a folder containing images.')
    argparser.add_argument('--save_path',
                           type=str,
                           default='',
                           help='folder to save the detection results')
    argparser.add_argument('--not_display',
                           action='store_true',
                           help='turn off displaying images while running '
                                'detection')

    args = argparser.parse_args()
    main(args)
