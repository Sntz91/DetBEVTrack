import math
import random
from src.object_detection.detector import Detector
from src.bev.position_estimator import PositionEstimator
import numpy as np

# TODO CONFIG
DISTANCE_THRESHOLD = 20
TIME_TO_DESTROY_THRESHOLD = -6



class Detection:
    def __init__(self, id, position, label, score):
        self.id = id
        self.position = position
        self.label = label
        self.score = score
        self.active_flg = True

    def belongs_to_object(self, obj):
        return self._calculate_distance(obj) < DISTANCE_THRESHOLD

    def _calculate_distance(self, obj):
        distance = math.hypot(
            obj.current_position[0] - self.position[0],
            obj.current_position[1] - self.position[1]
        )
        return distance



class Object:
    def __init__(self, id, position):
        self.id = id
        self.current_position = position
        self.position_history = []
        self.t_last_detection_match = 0
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.prediction = self.current_position


    def predict_next_position(self):
        if len(self.position_history) <= 1:
            return np.array(self.current_position)
        x = np.array(self.current_position)
        #print('x', x)
        x_prev = np.array(self.position_history[-1])
        #print('x_prev', x_prev)
        v = np.array([x[0] - x_prev[0], x[1] - x_prev[1]])
        # BUG v is 0 if detection 0. 
        #print('v', v)
        next_position = x + v
        #print('next_position', next_position)
        return next_position

    def __repr__(self):
        return f'id: {self.id}, current_position: {self.current_position}'

    def update_position(self, position):
        self.position_history.append(self.current_position)
        self.current_position = position
        self.prediction = self.predict_next_position().tolist()

    def add_detection(self, detection):
        self.update_position(detection.position)
        self.t_last_detection_match = 0

    def is_old(self):
        return self.t_last_detection_match < TIME_TO_DESTROY_THRESHOLD

    def has_not_been_updated(self):
        return self.t_last_detection_match < 0

    def set_prediction(self, x):
        self.prediction = x

    def append_to_history(self, x):
        self.position_history.append(x)

    def set_current_position(self, x):
        self.current_position = x

    def get_prediction(self):
        return self.prediction

    # wenn von letztes measurement von current prediction -> kill.
    # Alternativ:
    # Ende / Anfang von Bild -> kill (bestimmte world coordinate)



class Tracker:
    """
        Input: Detections
        Output: Objects 
    """
    def __init__(self):
        self.objects = []
        self.detections = []
        self.timestep = 0
        self.object_counter = 0

    @staticmethod
    def no_detections(labels):
        return len(labels) == 0

    def set_current_detections(self, bev_message):
        """
            Set current detections from OD/BEV Algorithm. 
        """
        ids, labels, scores, positions, hists = \
            PositionEstimator.unpack_message(bev_message)
        if self.no_detections(labels):
            return
        for label, score, position, hist in zip(labels, scores, positions, hists):
            new_detection = Detection(len(self.detections), position, label, score)
            self.detections.append(new_detection)

    def reset_detections(self):
        """
            Reset detections. After every tracking-step the detections should be resetted. 
        """
        self.detections = []

    def track(self, bev_message):
        """
            Iterate through all detections and objects,
            compare them and either:
            - add new objects
            - merge object and detection
            - delete old objects 
        """
        self.set_current_detections(bev_message)
        
        for detection in list(self.detections):
            #if self._no_object_created_yet: # why is this an error?
            if len(self.objects) == 0:
                self.add_new_object(detection)

            for obj in list(self.objects):
                if detection.belongs_to_object(obj):
                    self.merge_object_and_detection(detection, obj)
                    self.detections.remove(detection)
                if obj.is_old():
                    self.objects.remove(obj)

        for leftover_detection in self.detections:
            self.add_new_object(leftover_detection)

        # BUG It doesnt set the object to the next position if no detection found

        for obj in self.objects:
            if obj.has_not_been_updated():
                obj.update_position(obj.get_prediction())
                # The problem is, that the position has already been added to history.
                #print(obj.current_position, ' != ', obj.position_history)
                #print(obj.position_history[-1])
                #prediction = obj.predict_next_position()
                #obj.set_prediction(prediction)
                #print('prediction', obj.prediction)
                #print('append current_position to history', obj.current_position)
                #obj.append_to_history(obj.current_position)
                #obj.set_current_position(prediction)
                #print('current position is prediction', obj.current_position)
                #new_prediction = obj.predict_next_position()
                #obj.set_prediction(new_prediction)
            obj.t_last_detection_match -= 1

            #obj.t_last_detection_match -= 1
            #if obj.t_last_detection_match < -1:
            #    if obj.prediction is not None:
            #        obj.prediction = obj.predict_next_position()
            #        print('update with prediction')
            #        print('t', obj.current_position)
            #        print('t-1', obj.position_history[-1])
            #        print('t+1', obj.prediction)
            #        obj.update_position(obj.prediction)

        # if object has not been updated yet, then use prediction
         
        self.timestep += 1
        self.reset_detections()
        return self.pack_message()

    def _no_object_created_yet(self):
        return len(self.objects) == 0

    def pack_message(self):
        message = {}
        message["objects"] = []
        for obj in self.objects:
            message["objects"].append({
                "id": obj.id,
                "current_position": obj.current_position,
                "position_history": obj.position_history,
                "color": obj.color,
                "prediction": obj.prediction
            })
        return message

    @staticmethod
    def unpack_message(message):
        objects = message["objects"]
        if len(objects) == 0:
            return [], [], [], [], []
        ids, current_positions, position_histories, color, prediction = zip(*
            [[
                obj["id"],
                obj["current_position"],
                obj["position_history"],
                obj["color"],
                obj["prediction"]
            ] for obj in objects]
        )
        return ids, current_positions, position_histories, color, prediction

    def merge_object_and_detection(self, detection, obj):
        """
            If a detection matches an existing object, merge them. 
        """
        obj.add_detection(detection)

    def add_new_object(self, detection):
        """
            Add new detection, increment object counter (for ids) 
        """
        new_id = self.object_counter
        new_object = Object(new_id, detection.position)
        self.objects.append(new_object)
        self.object_counter += 1
    

