import cv2
import numpy as np
from utils import *
from image_stream import image_streamer


frame_id = 0

def calculate_control_signal(current_speed, image):
    global frame_id

    frame_id += 1
    # write_image("rgb", frame_id, image)

    steering_angle = 0
    left_point, right_point, im_center = find_lane_lines(image)
    if left_point != -1 and right_point != -1:

        # Calculate difference between car center point and image center point
        center_point = (right_point + left_point) // 2
        center_diff =  center_point - im_center

        # Calculate steering angle from center point difference
        steering_angle = float(center_diff * 0.04)

    # Constant throttle = 0.5 * MAX SPEED
    throttle = 0.5

    return throttle, steering_angle


def grayscale(img):
    """Chuyển ảnh màu sang ảnh xám"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def birdview_transform(img):

    IMAGE_H = 160
    IMAGE_W = 320

    src = np.float32([[0, IMAGE_H], [320, IMAGE_H], [0, IMAGE_H//3], [IMAGE_W, IMAGE_H//3]])
    dst = np.float32([[90, IMAGE_H], [230, IMAGE_H], [-10, 0], [IMAGE_W+10, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img


def preprocess(img):
    
    img = grayscale(img)
    image_streamer.set_image("grayscale", img)

    img = gaussian_blur(img, 11)
    image_streamer.set_image("gaussian_blur", img)

    img = canny(img, 150, 200)
    image_streamer.set_image("canny", img)

    img = birdview_transform(img)
    image_streamer.set_image("birdview", img)

    return img


def find_lane_lines(image, draw=False):

    image = preprocess(image)

    im_height, im_width = image.shape[:2]

    if draw: viz_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Vạch kẻ sử dụng để xác định tâm đường
    interested_line_y = int(im_height * 0.7)
    if draw: cv2.line(viz_img, (0, interested_line_y), (im_width, interested_line_y), (0, 0, 255), 2) 
    interested_line = image[interested_line_y, :]

    # Xác định điểm bên trái và bên phải
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
    
    # Dự đoán điểm trái và điểm phải nếu chỉ thấy 1 trong 2 điểm
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
