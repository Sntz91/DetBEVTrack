import numpy as np
import torch
import cv2
import json
from . import config as cfg
from scipy.spatial import ConvexHull
from dataclasses import dataclass

@dataclass
class Detection:
    id: id
    box: tuple
    label: str
    score: int
    mask: list
    color_hist: np.array
    position: np.array

    def pack_message(self):
        output = dict()
        output["id"] = self.id
        output["box"] = self.box
        output["label"] = self.label
        output["score"] = self.score
        output["mask"] = self.mask
        output["color_hist"] = self.color_hist
        output["position"] = self.position
        return output

    def unpack_message(self):
        pass

    def save_message(self):
        pass

# Wrapper for detections
class Detections:
    pass



CLASSES_OF_INTEREST = ['person', 'bycicle', 'car']


class Detector:
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    def _detect(self, frame: np.array, conf_threshold: float):
        detections = self.model(frame)
        boxes, labels, scores = Detector.convert_yolo_preds(
            detections.xyxy[0].data, conf_threshold
        )
        return boxes, labels, scores

    def detect(self, frame: np.array, masks: np.array, conf_threshold: float, ts: int, camera: str) -> tuple:
        boxes, labels, scores = self._detect(frame, conf_threshold)
        message = Detector.pack_message(boxes, labels, scores, masks)
        return message

    @staticmethod
    def get_xyxy_from_box(box):
        return tuple(int(coordinate) for coordinate in box)

    @staticmethod
    def convert_box_xyxy_to_xywh(box):
        x = int(box[0])
        y = int(box[1])
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        return [x,y,w,h]


    @staticmethod
    def cut_mask(masks, box):
        x, y, x2, y2 = Detector.get_xyxy_from_box(box)
        mask = masks[y:y2, x:x2, :]
        mask = Detector.convert_mask_to_hull(mask)
        return mask

    @staticmethod
    def convert_mask_to_hull(mask):
        return mask # TODO

    @staticmethod
    def pack_message(boxes, labels, scores, masks):
        message = {}
        message["detections"] = []
        for box, label, score in zip(boxes, labels, scores):
            mask = Detector.cut_mask(masks, box)
            message["detections"].append({
                "id": 1,
                "box": Detector.convert_box_xyxy_to_xywh(box),
                "label": label,
                "score": score,
                "mask": mask.tolist(),
                "color_hist": None
            })
        return message

    @staticmethod
    def unpack_message(message):
        detections = message["detections"]

        if len(detections) == 0:
            return [], [], [], [], [], []
        ids, boxes, labels, scores, masks, hists = zip(*
            [[
                obj["id"], 
                obj["box"], 
                obj["label"], 
                obj["score"], 
                np.array(obj["mask"], np.uint8), 
                obj["color_hist"]
             ] for obj in detections]
        )
        return ids, boxes, labels, scores, masks, hists


    @staticmethod
    def convert_yolo_preds(preds, conf_threshold):
        if len(preds) == 0:
            return [], [], []
        boxes = preds[:, :4]
        labels = preds[:, 5]
        scores = preds[:, 4]

        labels = [cfg.CLASSES_YOLO[int(label.item())] for label in labels]

        filtered_preds = [
            (
                Detector.convert_tensor_to_list(box),
                label, 
                Detector.convert_tensor_to_list(score)
            )
            for box, label, score in zip(boxes, labels, scores) 
            if label in cfg.CLASSES_OF_INTEREST
            and score >= conf_threshold]

        if len(filtered_preds) < 3:
            return [], [], []
        else:
            filtered_boxes, filtered_labels, filtered_scores = zip(*filtered_preds)
        return filtered_boxes, filtered_labels, filtered_scores
        
    @staticmethod
    def convert_tensor_to_list(x):
        return x.numpy().tolist()