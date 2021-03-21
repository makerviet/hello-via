import cv2
from pathlib import Path
import os
import base64

MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED

def get_speed_limit(speed):
    global speed_limit
    if speed > speed_limit:
        speed_limit = MIN_SPEED
    else:
        speed_limit = MAX_SPEED
    return speed_limit

def calculate_control_signal(left_point, right_point, im_center):
    global frame_id

    steering_angle = 0
    if left_point != -1 and right_point != -1:

        # Calculate difference between car center point and image center point
        center_point = (right_point + left_point) // 2
        center_diff =  center_point - im_center

        # Calculate steering angle from center point difference
        steering_angle = float(center_diff * 0.04)

    # Constant throttle = 0.5 * MAX SPEED
    throttle = 0.5

    return throttle, steering_angle

def send_control(sio, steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

