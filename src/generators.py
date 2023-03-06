import glob
import cv2
import os
import json

class CameraDataGenerator():
    def __init__(self, frame_dir, mask_dir, gt_filename):
        self.frame_generator = FrameGenerator(frame_dir)
        self.mask_generator = FrameGenerator(mask_dir)
        self.gt_generator = GTGenerator(gt_filename)
        assert len(self.frame_generator) \
            == len(self.mask_generator) \
            == len(self.gt_generator), \
            f"Lengths of generators not identical. \
                gt: {len(self.gt_generator)}, \
                masks: {len(self.mask_generator)}, \
                frames: {len(self.frame_generator)}"
        
    def __iter__(self):
        for frame, mask, gt in zip(self.frame_generator, self.mask_generator, self.gt_generator):
            yield frame, mask, gt

    def __len__(self):
        return len(self.frame_generator)


class CameraStreamGenerator():
    pass


class GTGenerator:
    def __init__(self, filename: str) -> None:
        self.positions = self.load_positions(filename)

    def load_positions(self, filename: str) -> list:
        positions = []
        assert os.path.exists(filename)
        with open(filename) as file:
            gt = json.load(file)
        for _, gt_positions in gt.items():
            positions_per_ts = []
            for _, position_per_obj in gt_positions.items():
                positions_per_ts.append(position_per_obj)
            positions.append(positions_per_ts)
        return positions
    
    def __iter__(self):
        for position in self.positions:
            yield position
    
    def __len__(self):
        return len(self.positions)


class FrameGenerator:
    def __init__(self, f_dir):
        self.filenames = sorted(glob.glob(f'{f_dir}/*.png'))
        self.file_cnt = 0
        self.maxlen = len(self.filenames)

    def __iter__(self):
        return self

    def __len__(self):
        return self.maxlen
    
    def __next__(self):
        cnt = self.file_cnt
        if self._is_finished(cnt):
            raise StopIteration
        self.file_cnt += 1 ##decorator??
        return cv2.imread(self.filenames[cnt])
    
    def _is_finished(self, cnt):
        return cnt >= self.maxlen


def test():
    camera_gen = CameraDataGenerator(
        'data/input/camera_1/frames', 
        'data/input/camera_2/masks', 
        'data/input/positions.json'
    )
    for frame, mask, gt in camera_gen:
        print(gt)

if __name__=='__main__':
    test()