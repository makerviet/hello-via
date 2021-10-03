#!/usr/bin/env python

import asyncio
import websockets
from PIL import Image
import json
import cv2
import numpy as np
import base64
from io import BytesIO
from controller import calculate_control_signal


count = 0
async def echo(websocket, path):
    global count
    async for message in websocket:
        count += 1
        # print(count)
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        # Calculate speed and steering angle
        throttle, steering_angle = calculate_control_signal(image.copy())
        message = json.dumps({"throttle": throttle, "steering": steering_angle})
        await websocket.send(message)
        

async def main():
    async with websockets.serve(echo, "localhost", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

asyncio.run(main())