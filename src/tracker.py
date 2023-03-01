#TO BE DONE - Replaces object_tracking folder
import math
import random
import numpy as np
from dataclasses import dataclass, field

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
        self.assigned = False

    def __repr__(self):
        return f'Detection pos: {self.position}, label: {self.label}, score: {self.score:.2f}'

    def belongs_to_object(self, obj):
        return self._calculate_distance(obj) < DISTANCE_THRESHOLD
    
    def already_assigned(self):
        return self.assigned 

    def _calculate_distance(self, obj):
        distance = math.hypot(
            obj.current_position[0] - self.position[0],
            obj.current_position[1] - self.position[1]
        )
        return distance


class TrackObject:
    def __init__(self, id, position, label):
        self.id = id
        self.label = label
        self.current_position = position
        self.position_history = []
        self.t_last_detection_match = 0
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.prediction = self.current_position

    def get_dict(self):
        output = dict()
        output['id'] = self.id
        output['label'] = self.label
        output['current_position'] = self.current_position
        output['position_history'] = self.position_history
        output['color'] = self.color
        output['prediction'] = self.prediction
        return output

    def predict_next_position(self):
        if len(self.position_history) <= 1:
            return np.array(self.current_position)
        x = np.array(self.current_position)
        x_prev = np.array(self.position_history[-1])
        v = np.array([x[0] - x_prev[0], x[1] - x_prev[1]])
        next_position = x + v
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


@dataclass
class Tracks:
    track_list: list[TrackObject] = field(default_factory=list)

    def __repr__(self):
        return f'Nr. of Tracks: {len(self.track_list)}'
    
    def __iter__(self):
        for track in self.track_list:
            yield track

    def __len__(self):
        return len(self.track_list)

    def append_track(self, track: TrackObject):
        self.track_list.append(track)

    def remove(self, obj):
        self.track_list.remove(obj)

    def flush_track(self):
        self.track_list = list()

    def get_dict(self):
        output = dict()
        output["tracks"] = []
        for track in self.track_list:
            output["tracks"].append(
                track.get_dict()
            )
        return output


class Tracker:
    """
        Input: Detections
        Output: Objects 
    """
    def __init__(self):
        self.tracks = Tracks()
        self.detections = []
        self.timestep = 0
        self.object_counter = 0

    def get_dict(self):
        message = dict()
        message["tracks"] = []
        for track in self.track_list:
            message["tracks"].append(
                track.get_dict()
            )
        return message
    

    def set_current_detections(self, positions):
        """
            Set current detections from OD/BEV Algorithm. 
        """
        for position in positions:
            new_detection = Detection(len(self.detections), position.position, position.label, position.score)
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

        det_cnt = 0 
        for detection in list(self.detections):
            det_cnt += 1
            #print(f'Start with detection {det_cnt} / {len(self.detections)+1}: ', detection)
            if len(self.tracks) == 0:
                #print(f'Edge Case: No Tracks yet. Add new object.')
                self.add_new_object(detection)
            
            obj_cnt = 0
            for obj in list(self.tracks):
                obj_cnt+=1 
                #print(f'Start comparing detection to objects {obj_cnt} / {len(self.tracks) + 1}: ', obj.id)
                if detection.belongs_to_object(obj):
                    #print('Detection belongs to object. Merging..')
                    if not detection.assigned:
                        self.merge_object_and_detection(detection, obj)
                        self.detections.remove(detection)
                    #else:
                    #    print('Detection already assigned.')
                if obj.is_old():
                    #print('Object is old. Remove it')
                    self.tracks.remove(obj)

        for leftover_detection in self.detections:
            #print('Add leftover detection as new object.')
            self.add_new_object(leftover_detection)


        for obj in self.tracks:
            if obj.has_not_been_updated():
                obj.update_position(obj.get_prediction())
            obj.t_last_detection_match -= 1

        self.timestep += 1
        self.reset_detections()
        return self.tracks

    def _no_object_created_yet(self):
        return len(self.tracks) == 0

    def merge_object_and_detection(self, detection, obj):
        """
            If a detection matches an existing object, merge them. 
        """
        obj.add_detection(detection)
        detection.assigned = True

    def add_new_object(self, detection):
        """
            Add new detection, increment object counter (for ids) 
        """
        new_id = self.object_counter
        new_object = TrackObject(new_id, detection.position, detection.label)
        self.tracks.append_track(new_object)
        detection.assigned = True
        self.object_counter += 1
    

