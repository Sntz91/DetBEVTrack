import numpy as np
import torch
from dataclasses import dataclass, field
import numpy as np

LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
    'teddy bear', 'hair drier', 'toothbrush'
] # TODO

@dataclass
class Detection:
    """ .tbd. """
    id = 1
    xyxy: tuple
    label: str
    score: int 
    mask: list = field(default_factory=list, repr=False)
    color_hist: np.array = field(default_factory=list, repr=False)

    def _set_mask(self, mask):
        self.mask = mask

    def _set_color_Hist(self, color_hist):
        self.color_hist = color_hist

    def get_dict(self):
        output = dict()
        output["id"] = self.id
        output["box"] = self.xywh()
        output["label"] = self.label
        output["score"] = self.score
        output["mask"] = self.mask.tolist()
        output["color_hist"] = self.color_hist
        return output

    def xywh(self):
        return [
            int(self.xyxy[0]), 
            int(self.xyxy[1]), 
            int(self.xyxy[2]) - int(self.xyxy[0]),
            int(self.xyxy[3]) - int(self.xyxy[1])
        ]

    def is_from_interest(self, conf_threshold, classes_of_interest):
        return ((self.label in classes_of_interest) & (self.score > conf_threshold))


@dataclass
class Detections:
    """ Wrapper Class for Detections """
    detection_list: list[Detection] = field(default_factory=list)

    def __repr__(self):
        return f'Nr. of detections: {len(self.detection_list)}'
    
    def __iter__(self):
        for detection in self.detection_list:
            yield detection


    def append_detection(self, detection: Detection):
        self.detection_list.append(detection)

    def set_masks(self, masks):
        for detection in self.detection_list:
            mask = self.cut_mask(masks, detection.xyxy)
            detection._set_mask(mask)

    def flush_detections(self):
        self.detection_list = list()
    
    def get_dict(self):
        output = dict()
        output["detections"] = []
        for detection in self.detection_list:
            output["detections"].append(
                detection.get_dict()
            )
        return output
    
    def cut_mask(self, masks, box):
        x, y, x2, y2 = (int(coordinate) for coordinate in box)
        conv_masks = convert_mask(masks)
        cut_mask = conv_masks[y:y2, x:x2, :]
        # Convert to Hull if necessary
        return cut_mask#convert_mask(mask)

def convert_mask(mask): # TODO
    new_mask = mask.copy()
    mask_r = mask[:,:,2].copy()
    new_mask[:,:,0] = (mask_r == 10) * 255
    new_mask[:,:,1] = (mask_r == 10) * 255
    new_mask[:,:,2] = (mask_r == 10) * 255
    return new_mask


class Detector:
    def __init__(self, conf_threshold=0.4, classes_of_interest=['car']) -> None: # TODO
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detections_per_ts = Detections()
        self.conf_threshold = conf_threshold# TODO
        self.classes_of_interest = classes_of_interest

    def _detect(self, frame: np.array):
        frame_detections = self.model(frame)
        # maybe i should outsource this too?
        for detection in frame_detections.xyxy[0]:
            detection_candidate = Detection(
                xyxy = [
                    detection[0].item(),
                    detection[1].item(),
                    detection[2].item(),
                    detection[3].item()
                ],
                score = detection[4].item(),
                label = LABELS[int(detection[5].item())] 
            )
            if detection_candidate.is_from_interest(self.conf_threshold, self.classes_of_interest):
                self.detections_per_ts.append_detection(detection_candidate)

    def detect(self, frame: np.array, masks: np.array) -> Detections: 
        self.detections_per_ts.flush_detections()
        self._detect(frame)
        self.detections_per_ts.set_masks(masks)
        return self.detections_per_ts
    