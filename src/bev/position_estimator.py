import numpy as np
import json
import datetime
from ..object_detection.detector import Detector
import cv2


class PositionEstimator:
    def __init__(self, visualize=False):
        self.visualize = visualize
        pass

    def estimate(self, detector_message, H, scale_factor, ts, camera, gt):
        ids, boxes, labels, scores, masks, hists = \
            Detector.unpack_message(detector_message)

        output = []

        self.gt = gt

        for id, box, label, score, mask, hist in zip(ids, boxes, labels, scores, masks, hists):
            # With Hull -> TODO funktioniert noch nicht richtig.
            #hull = PositionEstimator.cvt_mask_to_hull(mask)
            #min_rect_pts = self.get_min_rect_points(hull)

            instance_pts = self.iterate_img_and_find_car_pixels(mask)
            min_rect_pts = self.get_min_rect_points(instance_pts)

            min_rect_pts_image = self.move_ground_contact_points_by_bb_coordinates(min_rect_pts, box)
            ground_contact_points_image = self.find_ground_contact_line(min_rect_pts_image)
            ground_contact_points_world = self.transform_ground_contact_points_from_image_to_world(ground_contact_points_image, H)
            rotated_rvec = self.find_and_rotate_rvec_of_bottom_straight_by_degree(ground_contact_points_world, 90)
            ground_contact_point_world = self.calc_midpoint_from_two_points(ground_contact_points_world)
            shifted_ground_contact_point_world = self.shift_point_by_rvec_and_object_class(ground_contact_point_world, rotated_rvec, label, scale_factor)

            output_msg = self.pack_message(1, label, score, shifted_ground_contact_point_world, 0)
            output.append(output_msg)


        return output

    @staticmethod
    def draw_gt_positions(img, position_file):
        # PROBLEM: Positions is Carla positions...
        #img = cv2.imread('data/input/reference_image.png')
        with open(position_file) as f:
            positions = json.load(f)

        for ts, car in positions.items():
            for id, position in car.items():
                x, y = position
                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)

    @staticmethod
    def unpack_message(message):
        if len(message) == 0:
            return [], [], [], [], []
        ids, labels, scores, positions, hists = zip(*
            [[
                obj["id"], 
                obj["label"], 
                obj["score"], 
                obj["position"], 
                obj["color_hist"]
             ] for obj in message]
        )
        return ids, labels, scores, positions, hists

    def visualize_message(self, message, ts, camera):
        img = cv2.imread('data/input/reference_image.png')
        for obj in message:
            pt = np.int32(obj["position"])
            cv2.circle(img, (pt[0], pt[1]), 10, (0, 0, 255), -1)
        #img = self.draw_gt_positions(img, 'data/input/positions.json')
        for obj_gt in self.gt:
            x, y = int(obj_gt[0]), int(obj_gt[1])
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
        #ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        cv2.imwrite(f'data/visualizations/bev/{ts}_{camera}.png', img)
        PositionEstimator.save_message(message, ts, camera)

    @staticmethod
    def save_message(message, ts, camera):
        with open(f'data/visualizations/bev/messages/{ts}_{camera}.json', "w") as file:
            json.dump(message, file)

    @staticmethod
    def iterate_img_and_find_car_pixels(mask):
        img = mask.copy()
        width = img.shape[1]
        height = img.shape[0]
        i = 0
        car_pts = list()
        while i < height:
            j = 0
            while j < width:
                if(img[i, j][0] == 255):
                    car_pts.append([i, j])
                j += 1
            i += 1
        return car_pts
    

    @staticmethod
    def find_ground_contact_line(min_rect_pts):
        bottom_points, top_points, is_square = PositionEstimator.find_bottom_top_edge(min_rect_pts)
        return bottom_points

    @staticmethod 
    def get_min_rect_points(mask):
        point_array = np.array(mask)
        min_rect = cv2.minAreaRect(np.float32(point_array))
        min_rect_points = cv2.boxPoints(min_rect)
        angle_bb = min_rect[2]
        min_rect_points = np.int32(min_rect_points)
        return min_rect_points

    @staticmethod
    def pack_message(id, label, score, position, color_hist):
        d = dict()
        d["id"] = id
        d["label"] = label
        d["score"] = score
        d["position"] = [position[0].item(), position[1].item()] # Refactor
        d["color_hist"] = color_hist
        return d


    @staticmethod
    def shift_point_by_rvec_and_object_class(ground_contact_point_world, rotated_rvec, obj_class, scale_factor):
        point = ground_contact_point_world
        rvec = rotated_rvec
        if obj_class == "pedestrian":
            length = 0
        elif obj_class == "bicycle":
            length = 0
        elif obj_class == "car":
            length = 1.0
        elif obj_class == "van":
            length = 1.1
        elif obj_class == "truck":
            length = 1.5
        elif obj_class == "bus":
            length = 1.5
        else:
            raise Exception("Error. Object Class is not known by Module. Check Message from Object Detector!!!")
        length = scale_factor * length
        shifted_point_x = point[0] + rvec[0] * length
        shifted_point_y = point[1] + rvec[1] * length
        shifted_point = [shifted_point_x, shifted_point_y]
        return shifted_point

    @staticmethod 
    def calc_midpoint_from_two_points(ground_contact_points_world):
        pt1 = ground_contact_points_world[0]
        pt2 = ground_contact_points_world[1]
        new_x = (pt1[0] + pt2[0]) / 2
        new_y = (pt1[1] + pt2[1]) / 2
        point = [new_x, new_y]
        return point

    @staticmethod
    def find_and_rotate_rvec_of_bottom_straight_by_degree(ground_contact_points_world, rotation_angle):
        point1_straight = ground_contact_points_world[0]
        point2_straigth = ground_contact_points_world[1]
        rvec_straight = np.array([[point2_straigth[0] - point1_straight[0]],
                                            [point2_straigth[1] - point1_straight[1]]], dtype=np.float32)
        rvec_straight = np.array([[(point2_straigth[0] - point1_straight[0]) / np.linalg.norm(rvec_straight)],
                                            [(point2_straigth[1] - point1_straight[1]) / np.linalg.norm(rvec_straight)]], dtype=np.float32)
        if rvec_straight[0] < 0:
            rvec_straight[0] *= -1
            rvec_straight[1] *= -1

        rotation_angle = np.deg2rad(rotation_angle)
        rot_mat = np.array([[np.cos(rotation_angle), np.sin(rotation_angle)],
                           [-np.sin(rotation_angle), np.cos(rotation_angle)]], dtype=np.float32)
        
        rvec_straight_rotated = np.matmul(rot_mat, rvec_straight)
        return rvec_straight_rotated


    @staticmethod
    def transform_ground_contact_points_from_image_to_world(ground_points, H):
        warped_points_list = list()
        for point in ground_points:
            point_new = [point[1], point[0], 1]
            warped_point = np.matmul(H, point_new)
            scaling = 1 / warped_point[2]

            warped_point[1] = warped_point[1] * scaling
            warped_point[2] = warped_point[2] * scaling
            warped_point[0] = warped_point[0] * scaling
            warped_points_list.append(warped_point)
        np.asarray(warped_points_list, dtype=np.float32)
        return warped_points_list

    @staticmethod
    def move_ground_contact_points_by_bb_coordinates(ground_contact_point_image, box):
        for point in ground_contact_point_image:
            point[0] = point[0] + box[1] #- box[3]
            point[1] = point[1] + box[0]
        return ground_contact_point_image

    @staticmethod
    def find_bottom_top_edge(min_rect_pts):
        """
        Use the rotated bounding box to select two edges, the top and bottom edge of the vehicle
        In case of an squared bbox, this will be used as base plate
        """
        box = min_rect_pts
        #box = self.min_rect_points
        bottom_edge = np.zeros((2, 2))
        top_edge = np.zeros((2, 2))
        unique_x, _ = np.unique(box[:, 0], return_counts=True)
        unique_y, _ = np.unique(box[:, 1], return_counts=True)

        is_square = False

        if len(box) == 4 and len(unique_x) > 2 and len(unique_y) > 2:
            # Sort points by y coordinate
            sort_order = box[:, 0].argsort()[::-1]
            # Select highest y coordinate
            index_highest_y_coordinate = sort_order[0]
            bottom_vertice_candidate_0 = box[(index_highest_y_coordinate - 1) % 4]
            bottom_vertice_candidate_1 = box[(index_highest_y_coordinate + 1) % 4]
            top_edge[0] = box[(index_highest_y_coordinate + 2) % 4]

            sort_by_y_desc = box[sort_order]
            #  This point is guranted part of the bottom edge of a object
            bottom_vertice_0 = sort_by_y_desc[0]
            bottom_edge[0] = bottom_vertice_0
            candidate_length_0 = np.linalg.norm(bottom_vertice_0 - bottom_vertice_candidate_0)
            candidate_length_1 = np.linalg.norm(bottom_vertice_0 - bottom_vertice_candidate_1)
            if candidate_length_0 > candidate_length_1:
                bottom_edge[1] = bottom_vertice_candidate_0
                top_edge[1] = bottom_vertice_candidate_1
            else:
                bottom_edge[1] = bottom_vertice_candidate_1
                top_edge[1] = bottom_vertice_candidate_0
        else:
            # This case will be called if the rotated bbox is a square
            is_square = True

            sorted_box = box[np.lexsort((box[:, 0], box[:, 1]))][::-1]

            candidate_length_0 = np.linalg.norm(sorted_box[0] - sorted_box[1])
            candidate_length_1 = np.linalg.norm(sorted_box[1] - sorted_box[3])

            if candidate_length_0 < candidate_length_1:
                top_edge[0] = sorted_box[1]
                top_edge[1] = sorted_box[3]
                bottom_edge[0] = sorted_box[2]
                bottom_edge[1] = sorted_box[0]
            else:
                top_edge[0] = sorted_box[0]
                top_edge[1] = sorted_box[1]
                bottom_edge[0] = sorted_box[3]
                bottom_edge[1] = sorted_box[2]

            if len(unique_y) == 2:
                # Sort by y-coordinates
                sort_order = box[:, 1].argsort()[::-1]
                # select first
                index_highest_y_coordinate = sort_order[0]
                sort_by_y_desc = box[sort_order]
                #  This point is guaranteed part of the bottom edge of a object
                bottom_vertice_0 = sort_by_y_desc[0]
                vertice1 = sort_by_y_desc[1]
                vertice2 = sort_by_y_desc[2]
                vertice3 = sort_by_y_desc[3]
                if bottom_vertice_0[0] == vertice2[0]:
                    length_vector_1 = np.linalg.norm(vertice2 - bottom_vertice_0)
                else:
                    length_vector_1 = np.linalg.norm(vertice3 - bottom_vertice_0)

                length_y_vector = np.linalg.norm(vertice1 - bottom_vertice_0)

                if length_y_vector > length_vector_1:
                    if bottom_vertice_0[0] < vertice1[0]:
                        bottom_edge[0] = vertice1
                        bottom_edge[1] = bottom_vertice_0
                    else:
                        bottom_edge[0] = bottom_vertice_0
                        bottom_edge[1] = vertice1
                    top_edge[0] = vertice2
                    top_edge[1] = vertice3
                    is_square = False

        #if bottom_edge[0][0] >= bottom_edge[1][0] or bottom_edge[0][1] <= bottom_edge[1][1]:
        #    bottom_edge = bottom_edge[::-1]
        return np.asarray(bottom_edge, dtype=np.float32), np.asarray(top_edge, dtype=np.float32), is_square






    @staticmethod
    def cvt_mask_to_hull(mask):
        # Make mask binary (kind of)
        mask_r = mask[:,:,0]
        mask[:,:,0] = (mask_r > 100) * 255
        mask[:,:,1] = (mask_r > 100) * 255
        mask[:,:,2] = (mask_r > 100) * 255
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(mask_gray, 127, 255, 0)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = contours[0]
        hull = cv2.convexHull(cnt)
        return hull