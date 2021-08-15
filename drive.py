import base64
import threading
import time
from io import BytesIO

import cv2
import eventlet
import numpy as np
import websocket
from flask import Flask, jsonify, redirect, request, send_from_directory
from flask_socketio import SocketIO
from PIL import Image
from werkzeug import debug

from controller import *
from image_stream import image_streamer
from utils import *

eventlet.monkey_patch()

app = Flask(__name__, static_url_path='')
sio = SocketIO(app)


def on_message(ws, message):
    data = json.loads(message)
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_streamer.set_image("rgb", image)

    # Calculate speed and steering angle
    throttle, steering_angle = calculate_control_signal(image.copy())
    send_control(ws, steering_angle, throttle)


@app.route('/')
def homepage():
    return redirect("/web/index.html?t={}".format(time.time()))

@app.route('/web/<path:path>')
def send_web(path):
    return send_from_directory('web', path)

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

# Start info streaming thread
def info_thread_func(sio):
    global count
    while True:
        sio.sleep(0.05)
        frame = image_streamer.get_image(image_streamer.get_current_topic())
        sio.emit(
            'server2web',
            {
                'image': convert_image_to_jpeg(frame),
                'topic': image_streamer.get_current_topic()
            },
            skip_sid=True, broadcast=True)
info_thread = threading.Thread(target=info_thread_func, args=(sio,))
info_thread.setDaemon(True)
info_thread.start()


def websocket_client_func():
    global on_message
    def on_error(ws, error):
        print(error)
    ws = websocket.WebSocketApp("ws://127.0.0.1:4567/simulation",
    on_message=on_message,
    on_error=on_error)
    ws.run_forever()
websocket_client_thread = threading.Thread(target=websocket_client_func)
websocket_client_thread.setDaemon(True)
websocket_client_thread.start()

print("Starting server. Go to: http://localhost:8080")
sio.run(app, port=8080)
