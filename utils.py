import cv2
from pathlib import Path
import os

MAX_SPEED = 25
MIN_SPEED = 10
speed_limit = MAX_SPEED

LOG_PATH = "logs"

def init_log_dirs():
    """Initialize log folder
    """
    Path(LOG_PATH).mkdir(parents=True, exist_ok=True)

def write_image(image_type, frame_id, image):
    """Write image to log dir
    """
    log_dir = os.path.join(LOG_PATH, image_type)
    if not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    img_path = os.path.join(log_dir, "{}.png".format(frame_id))
    cv2.imwrite(img_path, image)

def get_speed_limit(speed):
    global speed_limit
    if speed > speed_limit:
        speed_limit = MIN_SPEED
    else:
        speed_limit = MAX_SPEED
    return speed_limit

def send_control(sio, steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)