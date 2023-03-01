import cv2
import sys
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from src.detector import Detections
from src.position_estimator import Positions
from src.tracker import Tracker, Tracks
from src.utils import Visualizer

@dataclass
class EventData(ABC):
    output_dir: str
    camera_name: str
    ts: int = field(init=False)
    def __post_init__(self):
        Visualizer.clear_directory(self.output_dir)

    @abstractmethod
    def update(self):
        """ update the state of EventData Object. """

    @abstractmethod
    def visualize(self, name):
        """ visualize the object in window. """

    @abstractmethod 
    def get_visualization(self):
        """ get visualization of object. """

    @abstractmethod
    def log(self):
        """ log the object. """

    @abstractmethod
    def save_image(self):
        """ save visualization of object as image to self.output_dir. """

    @abstractmethod
    def save_message(self):
        """ save object in form of message / dict to self.output_dir. """


@dataclass
class InputDataEventData(EventData):
    frame: np.ndarray = field(init=False)

    def update(self, 
        frame: np.ndarray, 
        ts: int
    ) -> None:
        self.frame = frame
        self.ts = ts

    def visualize(self) -> None:
        viz = self.get_visualization(self.frame)
        cv2.imshow('input_viz', viz)
        k = cv2.waitKey(1)
        if k == ord('q'):
            sys.exit(0)

    def log(self) -> None:
        logging.debug(f'ts: {self.ts}, input provided.')

    def save_image(self) -> None:
        viz = self.get_visualization(self.frame)
        Visualizer.save_image(self.output_dir, self.camera_name, viz, self.ts) 

    def save_message(self) -> None:
        pass

    def get_visualization(self, frame: np.ndarray) -> np.ndarray:
        return frame
    

@dataclass
class ObjectDetectionEventData(EventData):
    detections: Detections = field(init=False)
    frame: np.ndarray = field(init=False)

    def update(self, 
        detections: Detections, 
        frame: np.ndarray, 
        ts: int
    ) -> None:
        self.detections = detections
        self.frame = frame
        self.ts = ts

    def visualize(self) -> None:
        viz = self.get_visualization(self.detections, self.frame)
        cv2.imshow('object_detection_viz', viz)
        k = cv2.waitKey(1)
        if k == ord('q'):
            sys.exit(0)

    def log(self) -> None:
        logging.debug(f'ts: {self.ts}, {self.detections}')

    def save_image(self) -> None:
        viz = self.get_visualization(self.detections, self.frame)
        Visualizer.save_image(self.output_dir, self.camera_name, viz, self.ts) 

    def save_message(self) -> None:
        output = self.detections.get_dict()
        Visualizer.save_message(self.camera_name, self.output_dir, output, self.ts)

    def get_visualization(self,
        detections: Detections, 
        frame: np.ndarray
    ) -> np.ndarray:
        new_frame = frame.copy()
        for detection in detections:
            new_frame = Visualizer.draw_box(new_frame, detection.xywh(), detection.score, detection.label)
            new_frame = Visualizer.draw_mask_inside_box(new_frame, detection.xywh(), detection.mask.tolist())
        return new_frame


@dataclass
class PositionEstimatorEventData(EventData):
    estimated_positions: Positions = field(init=False)
    gt_positions: list = field(init=False, default_factory=list)
    reference_image_name: str

    def __post_init__(self):
        Visualizer.clear_directory(self.output_dir)
        self.reference_image = cv2.imread(self.reference_image_name)
    
    def update(self,
        estimated_positions: Positions,
        gt_positions: list,
        ts: int
    ) -> None:
        self.gt_positions = gt_positions
        self.estimated_positions = estimated_positions
        self.ts = ts

    def visualize(self) -> None:
        viz = self.get_visualization(self.estimated_positions, self.gt_positions, self.reference_image)
        cv2.imshow('position_estimation_viz', viz) # TODO this can be outsourced somehow TODO TODO 
        k = cv2.waitKey(1)
        if k == ord('q'): 
            sys.exit(0)

    def get_visualization(self,
        estimated_positions: Positions, 
        gt_positions: list,
        reference_image: np.ndarray
    ) -> None:
        output_image = reference_image.copy()
        # two loops, because not every object may be detected.
        for position in estimated_positions:
            output_image = Visualizer.draw_circle(output_image, position.position, color=(0,0,255))
        for gt_position in gt_positions:
            output_image = Visualizer.draw_circle(output_image, gt_position, size=7)
        return output_image
    
    def log(self):
        pass

    def save_image(self): 
        viz = self.get_visualization(self.estimated_positions, self.gt_positions, self.reference_image)
        Visualizer.save_image(self.output_dir, self.camera_name, viz, self.ts) 

    def save_message(self):
        output = self.estimated_positions.get_dict()
        Visualizer.save_message(self.camera_name, self.output_dir, output, self.ts)


@dataclass
class TrackerEventData(EventData):
    tracks: Tracks = field(init=False)
    reference_image_name: str

    def __post_init__(self):
        Visualizer.clear_directory(self.output_dir)
        self.reference_image = cv2.imread(self.reference_image_name)

    def update(self,
        tracks: Tracker,
        ts: int
    ) -> None:
        self.tracks = tracks
        self.ts = ts

    def visualize(self):
        pass

    def get_visualization(self,
        tracks: Tracks,
        reference_image: np.array
    ) -> None:
        output_image = reference_image.copy()
        for track in tracks:
            output_image = Visualizer.draw_circle(output_image, track.prediction, color=(0, 255, 0), size=15)
            output_image = Visualizer.draw_circle(output_image, track.current_position, color=track.color, size=12)
            for history in track.position_history:
                output_image = Visualizer.draw_circle(output_image, history, color=track.color)
        return output_image

    def log(self):
        pass

    def save_image(self):
        viz = self.get_visualization(self.tracks, self.gt_positions, self.reference_image)
        Visualizer.save_image(self.output_dir, self.camera_name, viz, self.ts)

    def save_message(self):
        output = self.tracks.get_dict()
        Visualizer.save_message(self.camera_name, self.output_dir, output, self.ts)