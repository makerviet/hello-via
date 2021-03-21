import cv2
import numpy as np
from ..utils.image_stream import image_streamer
from ..utils.image_processing import *

class LaneDetector():

    def __init__(self, use_deep_learning=False, lane_segmentation_model=None):
        if use_deep_learning and lane_segmentation_model is None:
            print("Deep learning mode is on. Please input a lane line segmentation model.")
            exit(1)
        self.use_deep_learning = use_deep_learning
        self.lane_segmentation_model = lane_segmentation_model

    def find_lane_line_mask(self, img, use_deep_learning=False):
        """Get lane line mask using image processing or deep learning"""

        lane_mask = None
        if use_deep_learning:
            lane_mask = self.lane_segmentation_model.predict(img)
        else:
            img = grayscale(img)
            image_streamer.set_image("grayscale", img)

            img = gaussian_blur(img, 11)
            image_streamer.set_image("gaussian_blur", img)

            img = canny(img, 150, 200)
            image_streamer.set_image("canny", img)

            lane_mask = img

        birdview_img = birdview_transform(lane_mask)
        image_streamer.set_image("lane_segmentation/birdview", birdview_img)

        return img


    def find_lane_lines(self, image, draw=False, use_deep_learning=False):
        """Find lane lines from image"""

        image = self.find_lane_line_mask(image, use_deep_learning=use_deep_learning)
        im_height, im_width = image.shape[:2]

        if draw: viz_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Horizontal line for center detection
        interested_line_y = int(im_height * 0.7)
        if draw: cv2.line(viz_img, (0, interested_line_y), (im_width, interested_line_y), (0, 0, 255), 2) 
        interested_line = image[interested_line_y, :]

        # Determine left point and right point
        left_point = -1
        right_point = -1
        lane_width = 100

        center = im_width // 2
        for x in range(center, 0, -1):
            if interested_line[x] > 0:
                left_point = x
                break
        for x in range(center + 1, im_width):
            if interested_line[x] > 0:
                right_point = x
                break
        
        # Regress 1 more point when only 1 point is available
        if left_point != -1 and right_point == -1:
            right_point = left_point + lane_width
        if left_point == -1 and right_point != -1:
            left_point = right_point - lane_width

        if draw:
            if left_point != -1:
                viz_img = cv2.circle(viz_img, (left_point, interested_line_y), 7, (255,255,0), -1)
            if right_point != -1:
                viz_img = cv2.circle(viz_img, (right_point, interested_line_y), 7, (0,255,0), -1)

        if draw:
            return left_point, right_point, center, viz_img
        else:
            return left_point, right_point, center
