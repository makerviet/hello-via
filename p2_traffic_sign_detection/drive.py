import asyncio
import base64
import json
import time
from io import BytesIO
from multiprocessing import Process, Queue

import cv2
import numpy as np
import websockets
from PIL import Image

from lane_line_detection import *
from traffic_sign_detection import *

# Initalize traffic sign classifier
traffic_sign_model = cv2.dnn.readNetFromONNX(
    "traffic_sign_classifier_lenet_v3.onnx")

# Global queue to save current image
# We need to run the sign classification model in a separate process
# Use this queue as an intermediate place to exchange images
g_image_queue = Queue(maxsize=5)

# Function to run sign classification model continuously
# We will start a new process for this
def process_traffic_sign_loop(g_image_queue):
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue
        image = g_image_queue.get()

        # Prepare visualization image
        draw = image.copy()
        # Detect traffic signs
        detect_traffic_signs(image, traffic_sign_model, draw=draw)
        # Show the result to a window
        cv2.imshow("Traffic signs", draw)
        cv2.waitKey(1)


async def process_image(websocket, path):
    async for message in websocket:
        # Get image from simulation
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Prepare visualization image
        draw = image.copy()

        # Send back throttle and steering angle
        throttle, steering_angle = calculate_control_signal(image, draw=draw)

        # Update image to g_image_queue - used to run sign detection
        if not g_image_queue.full():
            g_image_queue.put(image)

        # Show the result to a window
        cv2.imshow("Result", draw)
        cv2.waitKey(1)

        # Send back throttle and steering angle
        message = json.dumps(
            {"throttle": throttle, "steering": steering_angle})
        await websocket.send(message)


async def main():
    async with websockets.serve(process_image, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    p = Process(target=process_traffic_sign_loop, args=(g_image_queue,))
    p.start()
    asyncio.run(main())
