import sys
sys.path.append("src/traffic_sign_detection/yolov5")

import cv2
import numpy as np
import torch
from numpy import random
from src.traffic_sign_detection.yolov5.models.experimental import attempt_load
from src.traffic_sign_detection.yolov5.utils.general import (
    apply_classifier, check_img_size, non_max_suppression, scale_coords,
    set_logging, xyxy2xywh)
from src.traffic_sign_detection.yolov5.utils.plots import plot_one_box
from src.traffic_sign_detection.yolov5.utils.torch_utils import (
    load_classifier, select_device)


class TrafficSignDetector:

    def __init__(self, model_path, use_gpu=False, image_size=224, conf_thres=0.5, iou_thres=0.5):

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.agnostic_nms = False

        self.weights = [model_path]

        set_logging()
        self.device = select_device("cpu" if not use_gpu else "0")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(
            [model_path], map_location=self.device)  # load FP32 model
        self.img_size = check_img_size(
            image_size, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

        # Run inference
        img = torch.zeros((1, 3, self.img_size, self.img_size),
                          device=self.device)  # init img
        _ = self.model(img.half(
        ) if self.half else img) if self.device.type != 'cpu' else None  # run once

    def predict(self, img0, visualize=False):

        # Padded resize
        img = TrafficSignDetector.letterbox(img0, new_shape=self.img_size)[0]

        # Prepare image as net input
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, agnostic=self.agnostic_nms)

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, img0)

        # Process detections
        draw = img0.copy()
        for i, det in enumerate(pred):  # detections per image
            # normalization gain whwh
            gn = torch.tensor(draw.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to draw size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], draw.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                            gn).view(-1).tolist()  # normalized xywh
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, draw, label=label,
                                 color=self.colors[int(cls)], line_thickness=3)


        # TODO: return predictions
        return [], draw

    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
        # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # only scale down, do not scale up (for better test mAP)
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / \
                shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)
