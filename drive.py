import base64
import threading
from io import BytesIO
import time

import cv2
import eventlet
import numpy as np
from flask import (Flask, redirect, request,
                   send_from_directory, jsonify)
from flask_socketio import SocketIO
from PIL import Image

from src.utils.image_stream import image_streamer
from src.utils.image_processing import image_to_jpeg_base64
from src.lane_detection.lane_detector import LaneDetector
from src.lane_detection.segmentation_model import LaneLineSegmentationModel
from src.control import calculate_control_signal, send_control
from src.traffic_sign_detection.traffic_sign_detector import TrafficSignDetector

eventlet.monkey_patch()

app = Flask(__name__, static_url_path='')
sio = SocketIO(app)

lane_line_model = LaneLineSegmentationModel("models/goodgame_laneline_pspnet_1712_0055.pb", use_gpu=False)
lane_detector = LaneDetector(use_deep_learning=True, lane_segmentation_model=lane_line_model)

traffic_sign_detector = TrafficSignDetector("models/via_traffic_sign_detection_20210321.pt", use_gpu=False)

@sio.on('telemetry')
def telemetry(data):
    global debug_images

    if data:
        throttle = float(data["throttle"])
        steering_angle = float(data["steering_angle"])
        current_speed = float(data["speed"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_streamer.set_image("rgb", image)

            # TODO: Better performance. By multithreading ?
            # Traffic sign detection
            # traffic_sign_detector.predict(image)

            # Analyze image
            left_point, right_point, im_center = lane_detector.find_lane_lines(image)

            # Calculate speed and steering angle
            throttle, steering_angle = calculate_control_signal(left_point, right_point, im_center)

            send_control(sio, steering_angle, throttle)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)


@app.route('/')
def homepage():
    return redirect("/web/index.html?t={}".format(time.time()))

@app.route('/web/<path:path>')
def send_web(path):
    return send_from_directory('src/web', path)

@sio.on('connect')
def connect():
    send_control(sio, 0, 0)
    print('[INFO] Client connected: {}'.format(request.sid))

@sio.on('disconnect')
def disconnect():
    print('[INFO] Client disconnected: {}'.format(request.sid))

@app.route('/api/get_topics')
def get_topics():
    return jsonify({
        "success": True,
        "topics": image_streamer.get_topics()
    })

@app.route('/api/set_topic')
def set_topic():
    topic = request.args.get("topic", "")
    ret, message = image_streamer.set_current_topic(topic)
    if ret:
        return jsonify({
            "success": True,
            "new_topic": topic
        })
    else:
        return jsonify({
            "success": False,
            "message": message
        })

def info_thread_func(sio):
    global count
    while True:
        sio.sleep(0.05)
        frame = image_streamer.get_image(image_streamer.get_current_topic())
        sio.emit(
            'server2web',
            {
                'image': image_to_jpeg_base64(frame),
                'topic': image_streamer.get_current_topic()
            },
            skip_sid=True, broadcast=True)

# Start info streaming thread
info_thread = threading.Thread(target=info_thread_func, args=(sio,))
info_thread.setDaemon(True)
info_thread.start()

print("Starting server. Go to: http://localhost:4567")
sio.run(app, port=4567, debug=True)
