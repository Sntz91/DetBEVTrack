import numpy as np
import cv2
import os
import glob
import json


class DataLoader:
    """
        Loads camera-files of input directory. For each camera, there has
        to be a own  directory, including two subdirectories "frames", "masks".
        Additionally, it needs to store a file named homography_matrix.json
    """
    def __init__(self, input_dir: str, skip=0) -> None:
        self.camera_dict = self._load_camera_dict(input_dir)
        self.gt_positions = np.array(self._load_gt(input_dir)) 
        self.reference_img = cv2.imread(os.path.join(input_dir, 'reference_image.png'))
        self.skip = skip

    def __len__(self) -> int:
        return self.gt_positions.shape[0]

    @staticmethod
    def _load_H_file(camera_dir, filename):
        H_file = os.path.join(camera_dir, 'homography_matrix.json')
        assert os.path.exists(H_file), "No file found."
        with open(H_file, 'r') as file:
            H = json.load(file)
        return H

    @staticmethod
    def _get_camera_name(camera_dir):
        return os.path.basename(camera_dir[:-1])

    @staticmethod
    def _get_frame_files(camera_dir):
        return sorted(glob.glob(f'{camera_dir}frames/*.png'))

    @staticmethod
    def _get_mask_files(camera_dir):
        return sorted(glob.glob(f'{camera_dir}masks/*.png'))

    @staticmethod
    def _get_camera_directories(i_dir):
        return glob.glob(f'{i_dir}/*/')

    def _load_camera_dict(self, i_dir: str) -> dict:
        camera_dict = dict()
        for camera_dir in self._get_camera_directories(i_dir):
            camera_name = self._get_camera_name(camera_dir)
            frame_files = self._get_frame_files(camera_dir)
            mask_files = self._get_mask_files(camera_dir)
            H = self._load_H_file(camera_dir, 'homography_matrix.json')

            camera_dict[camera_name] = {
                "frame_files": frame_files,
                "mask_files": mask_files,
                "H": H
            }
        return camera_dict

    def _load_gt(self, i_dir):
        positions = []
        gt_file = os.path.join(i_dir, 'positions.json')
        with open(gt_file) as file:
            gt = json.load(file)
        for _, gt_positions in gt.items():
            positions_per_ts = []
            for _, position_per_obj in gt_positions.items():
                positions_per_ts.append(position_per_obj)
            positions.append(positions_per_ts)
        return positions

    def _iterate(self, ts):
        try:
            masks = []
            frames = []
            Hs = []
            camera_names = []
            for camera_name, data in self.camera_dict.items():
                frame = cv2.imread(data["frame_files"][ts])
                mask = cv2.imread(data["mask_files"][ts])
                H = data["H"]

                camera_names.append(camera_name)
                frames.append(frame)
                masks.append(mask)
                Hs.append(H)
            gt = self.gt_positions[ts]
        except IOError as e:
            print("Error: Could not read file", e)
        else:
            yield (ts, np.array(camera_names), np.array(frames), np.array(masks), np.array(Hs), gt)
        finally:
            pass

    @staticmethod
    def stack_3_img(one: np.array, two: np.array, three: np.array) -> np.array:
        h, w, c = three.shape
        w = int(0.5*w)
        output = np.vstack((one, two))
        output = cv2.resize(output, dsize=(w, h))
        output = np.hstack((output, three))
        return output, w

    def load_data(self):
        for i in range(len(self)-self.skip): 
            yield from self._iterate(i+self.skip)


    def get_camera_names(self):
        cameras = []
        for camera_name, camera_data in self.camera_dict.items():
            cameras.append(camera_name)
        return cameras



if __name__=="__main__":
    loader = DataLoader('data/input', visualize=True)
    data = loader.load_data()
    for ts, camera_name, frames, masks, Hs, gt in data:
        print(ts)