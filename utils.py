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

def send_control(sio, steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

def convert_image_to_jpeg(image):
    frame = cv2.imencode('.jpg', image)[1].tobytes()
    frame = base64.b64encode(frame).decode('utf-8')
    return "data:image/jpeg;base64,{}".format(frame)