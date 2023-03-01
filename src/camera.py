from abc import ABC, abstractmethod
from tqdm import tqdm
from src.events import InputDataEventData, ObjectDetectionEventData, PositionEstimatorEventData, TrackerEventData
from src.generators import CameraDataGenerator
from src.detector import Detector
from src.position_estimator import PositionEstimator
from src.tracker import Tracker
import os
import json
import numpy as np

def visualize(log, save_image, save_message, viz):
    def decorator(func):
        def wrapper(*args, **kwargs):
            event_data_object = func(*args, **kwargs)
            if log:
                event_data_object.log()
            if save_image:
                event_data_object.save_image()
            if save_message:
                event_data_object.save_message()
            if viz:
                event_data_object.visualize()
            return True
        return wrapper
    return decorator


class Camera(ABC):
    def __init__(self, 
        name: str, 
        homography_matrix_filename: str,
        output_dir: str,
        reference_image_filename: str
    ) -> None:
        self.detector = Detector()
        self.position_estimator = PositionEstimator()
        self.tracker = Tracker()
        self.name = name
        self.homography_matrix = self._load_homography_matrix(homography_matrix_filename)
        self.inv_homography_matrix = self.invert_homography_matrix(self.homography_matrix)
        self.data_generator = self.get_data_generator()
        self.ts = 0
        self.output_dir = output_dir

        # maybe even params 
        self._input_event_data = InputDataEventData(
            camera_name = self.name,
            output_dir = f'{self.output_dir}/input'
        )
        self._detection_event_data = ObjectDetectionEventData(
            camera_name = self.name,
            output_dir = f'{self.output_dir}/object_detection'
        )
        self._position_estimation_event_data = PositionEstimatorEventData(
            camera_name = self.name,
            output_dir = f'{self.output_dir}/position_estimation',
            reference_image_name=reference_image_filename
        )
        self._track_event_data = TrackerEventData(
            camera_name = self.name,
            output_dir = f'{self.output_dir}/tracks',
            reference_image_name=reference_image_filename
        )

    @staticmethod 
    def _load_homography_matrix(filename):
        assert os.path.exists(filename), "No homography file found."
        with open(filename) as file:
            homography_matrix = json.load(file)
        return homography_matrix
    
    @staticmethod
    def invert_homography_matrix(homography_matrx):
        return np.linalg.inv(homography_matrx)
        

    @visualize(log=True, save_image=True, save_message=True, viz=False)
    def _input_event(self, ts, frame):
        self._input_event_data.update(
            frame=frame,
            ts=ts
        )
        return self._input_event_data

    @visualize(log=True, save_image=True, save_message=True, viz=False)
    def _detection_event(self, ts, frame, mask): 
        detections = self.detector.detect(frame, mask)
        self._detection_event_data.update(
            detections=detections,
            frame=frame,
            ts=ts
        )
        return self._detection_event_data

    @visualize(log=True, save_image=True, save_message=True, viz=False) 
    def _position_estimation_event(self, ts, detections, homography_matrix, inv_homography_matrix, scale_factor, gt):
        bev_positions = self.position_estimator.estimate(detections, homography_matrix, inv_homography_matrix, scale_factor)
        self._position_estimation_event_data.update(
            estimated_positions=bev_positions,
            gt_positions=gt,
            ts=ts
        )
        return self._position_estimation_event_data

    @visualize(log=True, save_image=True, save_message=True, viz=True) 
    def _track_event(self, ts, positions):
        tracks = self.tracker.track(positions)
        self._track_event_data.update(
            tracks=tracks,
            ts=ts
        )
        return self._track_event_data

    def run(self, skip=0):
        for frame, mask, gt in tqdm(self.data_generator, total=len(self.data_generator)):
            self.ts += 1
            if skip >= self.ts:
                continue
            self._input_event(self.ts, frame) 
            self._detection_event(self.ts, frame, mask) 
            self._position_estimation_event(
                ts = self.ts, 
                detections = self._detection_event_data.detections, 
                homography_matrix = self.homography_matrix, 
                inv_homography_matrix = self.inv_homography_matrix, 
                scale_factor = 81/5,
                gt = gt
            )
            self._track_event(
                ts = self.ts, 
                positions = self._position_estimation_event_data.estimated_positions
            )

    @abstractmethod
    def get_data_generator(self):
        """returns generator for camera data"""


class GeneratedFrameCamera(Camera):
    def __init__(self, 
        name: str,
        homography_matrix_filename: str,
        output_dir: str,
        frame_dir: str,
        mask_dir: str,
        gt_filename: str,
        reference_image_filename: str
    ) -> None:
        self.frame_dir = frame_dir
        self.mask_dir = mask_dir
        self.gt_filename = gt_filename
        super().__init__(name, homography_matrix_filename, output_dir, reference_image_filename)


    def get_data_generator(self):
        camera_generator = CameraDataGenerator(
            frame_dir=self.frame_dir,
            mask_dir=self.mask_dir,
            gt_filename=self.gt_filename
        )
        return camera_generator
    

class StreamCamera(Camera):
    def __init__(self):
        pass
