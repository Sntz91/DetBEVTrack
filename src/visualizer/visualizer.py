import numpy as np
import cv2
import json
import os
import sys
from ..object_detection.detector import Detector
from ..bev.position_estimator import PositionEstimator
from ..object_tracking.tracker import Tracker

class Visualizer(object):
    def __init__(self, path_to_ref_img, output_dir):
        #assert Visualizer.directory_exists(output_dir)
        self.clear_directory_with_confirmation(output_dir)
        assert Visualizer.file_exists(path_to_ref_img)

        self.reference_img = cv2.imread(path_to_ref_img)
        self.output_dir = output_dir

    @staticmethod
    def stack_3_img(one: np.array, two: np.array, three: np.array) -> np.array:
        h, w, c = three.shape
        w = int(0.5*w)
        output = np.vstack((one, two))
        output = cv2.resize(output, dsize=(w, h))
        output = np.hstack((output, three))
        return output, w

    def visualize_message(self, message, ts, camera, **args):
        pass

    @staticmethod
    def draw_circle_for_each_position(img: np.array, positions: list, 
        color: tuple=(0, 255, 0), size: int=10) -> np.array:
        if positions.shape[0] == 0:
            return 
        for position in positions:
            x, y = int(position[0]), int(position[1])
            cv2.circle(img, (x, y), size, color, -1)
        return img

    @staticmethod
    def draw_current_ts(img, ts, position=(20, 30)):
        cv2.putText(
            img, f'ts: {ts}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (255, 255, 255), 2
        )
        return img

    @staticmethod
    def clear_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            for item in os.listdir(directory):
                s = os.path.join(directory, item)
                if os.path.isfile(s):
                    os.remove(s)
                elif os.path.isdir(s):
                    Visualizer.clear_directory(s)

    @staticmethod
    def clear_directory_with_confirmation(directory):
        if os.path.exists(directory) and os.listdir(directory):
            confirm = input(f"Are you sure you want to delete all files in {directory}? (y/n)")
            if confirm == 'y':
                Visualizer.clear_directory(directory)
            else:
                print(f"{directory} remains unchanged. Exit")
                raise SystemExit
        else:
            Visualizer.clear_directory(directory)

    @staticmethod
    def directory_exists(directory):
        return os.path.exists(directory) and os.path.isdir(directory)
    
    @staticmethod
    def file_exists(file_path):
        return os.path.exists(file_path) and os.path.isfile(file_path)

    def save_message(self, message, ts, camera_name):
        filename = f'{self.output_dir}/messages/{ts}_{camera_name}.json'
        with open(filename, "w") as file:
            json.dump(message, file)

    def save_image(self, img, ts, camera_name):
        return cv2.imwrite(f'{self.output_dir}/{ts}_{camera_name}.png', img)

    def draw_mask_inside_box(self, img, box, mask):
        new_img = img.copy()
        x, y, w, h = box
        new_img[y:y+h, x:x+w, :] = mask
        return new_img

    def draw_box(self, img, box, score, label, color=(255, 0, 0)):
        new_img = img.copy()
        x, y, w, h = box
        cv2.rectangle(new_img, (x,y), (x+w,y+h), color, 2)
        cv2.putText(
            new_img, f"{label}: {score: .2f}", 
            (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, color, 2
        )
        return new_img


class DetectionVisualizer(Visualizer):
    def visualize_message(self, frame, message, ts, camera_name, original_masks):
        new_frame = frame.copy()
        ids, boxes, labels, scores, masks, hists = \
            Detector.unpack_message(message)
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            new_frame = self.draw_box(new_frame, box, score, label)
            new_frame = self.draw_mask_inside_box(new_frame, box, mask)
        new_frame, x_offset = Visualizer.stack_3_img(original_masks, frame, new_frame)
        self.save_image(new_frame, ts, camera_name)
        self.save_message(message, ts, camera_name)

class InputVisualizer(Visualizer):
    def visualize_message(self, frame, mask, gt_positions, ts, camera_name):
        img = self.reference_img.copy()
        img = self.draw_circle_for_each_position(img, gt_positions)
        if img is None:
            return
        img, x_offset = self.stack_3_img(frame, mask, img)
        img = self.draw_current_ts(img, ts)
        self.save_image(img, ts, camera_name)


class BevVisualizer(Visualizer):
    def visualize_message(self, frame, message, ts, camera_name, gt_positions, detector_msg):
        detector_frame = frame.copy()
        ref_img = self.reference_img.copy()
        ref_img = self.draw_circle_for_each_position(ref_img, gt_positions)
        if len(message) > 0:
            ids, labels, scores, positions, hists = \
                PositionEstimator.unpack_message(message)
            ref_img = self.draw_circle_for_each_position(ref_img, np.array(positions), color=(0, 0, 255)) # TODO np.array stuff
        # TODO more functions...
        ids, boxes, labels, scores, masks, hists = \
            Detector.unpack_message(detector_msg)
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            detector_frame = self.draw_box(detector_frame, box, score, label)
            detector_frame = self.draw_mask_inside_box(detector_frame, box, mask)

        ref_img, x_offset = Visualizer.stack_3_img(frame, detector_frame, ref_img)
        self.save_image(ref_img, ts, camera_name)
        self.save_message(message, ts, camera_name)


class TrackVisualizer(Visualizer):
    def __init__(self, path_to_ref_img, output_dir, live=False):
        super().__init__(path_to_ref_img, output_dir)
        self.live = live

    def visualize_message(self, message, ts, camera_name, bev_message, gt_positions, frame, detector_msg):
        detector_frame = frame.copy()
        ref_img = self.reference_img.copy()
        bev_img = self.reference_img.copy()
        ids, current_positions, position_histories, colors, predicted_positions = Tracker.unpack_message(message)
        for id_, current_position, position_history, color, predicted_position in zip(ids, current_positions, position_histories, colors, predicted_positions):
            ref_img = cv2.circle(ref_img, (int(predicted_position[0]), int(predicted_position[1])), 20, (0, 255, 0), -1)
            ref_img = cv2.circle(ref_img, (int(current_position[0]), int(current_position[1])), 16, color, -1) # TODO FUCT
            for position in position_history:
                ref_img = cv2.circle(ref_img, (int(position[0]), int(position[1])), 8, self.weaken_color(color, 0.8), -1) # TODO FUNC

        # bev
        bev_img = self.draw_circle_for_each_position(bev_img, gt_positions)
        if len(bev_message) > 0:
            ids, labels, scores, positions, hists = \
                PositionEstimator.unpack_message(bev_message)
            bev_img = self.draw_circle_for_each_position(bev_img, np.array(positions), color=(0, 0, 255))


        # detector
        ids, boxes, labels, scores, masks, hists = \
            Detector.unpack_message(detector_msg)
        for box, label, score, mask in zip(boxes, labels, scores, masks):
            detector_frame = self.draw_box(detector_frame, box, score, label)
            detector_frame = self.draw_mask_inside_box(detector_frame, box, mask)

        ref_img, x_offset = Visualizer.stack_3_img(detector_frame, bev_img, ref_img)
        if self.live:
            cv2.imshow('output', ref_img)
            k = cv2.waitKey(1)
        self.save_image(ref_img, ts, camera_name)
        #self.save_message(message, ts, camera_name) # TODO


    @staticmethod
    def weaken_color(color, c):
        color = np.array(color)
        color = c * color
        return (color[0], color[1], color[2])