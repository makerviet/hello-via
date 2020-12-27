import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from io import BytesIO
import cv2
from utils import *
from controller import *

sio = socketio.Server()
init_log_dirs()

@sio.on('telemetry')
def telemetry(sid, data):

    if data:
        throttle = float(data["throttle"])
        steering_angle = float(data["steering_angle"])
        current_speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Calculate speed and steering angle
            throttle, steering_angle = calculate_control_signal(current_speed, image)

            send_control(sio, steering_angle, throttle)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    send_control(sio, 0, 0)

app = socketio.WSGIApp(sio)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
