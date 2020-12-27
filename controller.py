import cv2
from utils import *


frame_id = 0

def calculate_control_signal(current_speed, image):
    global frame_id

    frame_id += 1

    write_image("rgb", frame_id, image)

    steering_angle = 0
    speed_limit = get_speed_limit(current_speed)
    throttle = 1.0 - steering_angle**2 - (current_speed/speed_limit)**2
    throttle = 0

    return throttle, steering_angle