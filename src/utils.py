import os
import cv2
import json
import numpy as np
from abc import ABC
import glob
from PIL import Image
import imageio

class Visualizer(ABC):
    @staticmethod
    def directory_exists(directory: str):
        return os.path.exists(directory) and os.path.isdir(directory)
    
    @staticmethod
    def stack_3_img(one: np.array, two: np.array, three: np.array) -> np.array:
        h, w, c = three.shape
        w = int(0.5*w)
        output = np.vstack((one, two))
        output = cv2.resize(output, dsize=(w, h))
        output = np.hstack((output, three))
        return output, w

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
    def draw_box(
        img: np.ndarray, 
        box: list, 
        score: float, 
        label: str, 
        color: tuple = (255, 0, 0)
    ) -> np.ndarray:
        new_img = img.copy()
        x, y, w, h = box
        cv2.rectangle(
            new_img, 
            (x,y), 
            (x+w,y+h), 
            color, 
            2
        )
        cv2.putText(
            new_img, 
            f"{label}: {score: .2f}", 
            (x,y-5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.2, 
            color, 
            2
        )
        return new_img

    @staticmethod
    def draw_mask_inside_box(
        img: np.ndarray,
        box: list, 
        mask: list
    ) -> np.ndarray:
        new_img = img.copy()
        x, y, w, h = box
        new_img[y:y+h, x:x+w, :] = mask
        return new_img
    
    @staticmethod
    def draw_circle(
        img: np.ndarray,
        position: list,
        size: int = 10,
        color: tuple = (0, 255, 0)
    ) -> np.ndarray:
        output = img.copy()
        x, y = int(position[0]), int(position[1])
        cv2.circle(output, (x, y), size, color, -1)
        return output
        

    @staticmethod 
    def save_image(
        output_dir: str,
        camera_name: str, 
        img: np.ndarray, 
        ts: int
    ) -> bool:
        dir_name = f'{output_dir}/{camera_name}'
        if not Visualizer.directory_exists(dir_name):
            os.mkdir(dir_name)
        return cv2.imwrite(f'{dir_name}/{ts:03d}.png', img)

    @staticmethod
    def save_message(
        camera_name: str,
        output_dir: str,
        message: dict,
        ts: int
    ) -> None:
        dir_name = f'{output_dir}/{camera_name}/messages'
        if not Visualizer.directory_exists(dir_name):
            os.mkdir(dir_name)
        filename = f'{dir_name}/{ts}.json'
        with open(filename, "w") as file:
            json.dump(message, file)


# TODO refactor pls
def create_gif(input_dir, object_detection_dir, position_estimation_dir, output_dir):
    assert os.path.exists(input_dir)
    assert os.path.exists(object_detection_dir)
    assert os.path.exists(position_estimation_dir)
    input_filenames = sorted(glob.glob(f'{input_dir}/*.png'))
    od_filenames = sorted(glob.glob(f'{object_detection_dir}/*.png'))
    bev_filenames = sorted(glob.glob(f'{position_estimation_dir}/*.png'))
    i = 0

    frames = []

    for f1, f2, f3 in zip(input_filenames, od_filenames, bev_filenames):
        input_img = cv2.imread(f1)
        od_img = cv2.imread(f2)
        bev_img = cv2.imread(f3)
        
        output_image, _ = Visualizer.stack_3_img(input_img, od_img, bev_img)
        output_image = cv2.putText(output_image, f'ts: {i}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        #scale_percent = 30 # percent of original size
        #width = int(output_image.shape[1] * scale_percent / 100)
        #height = int(output_image.shape[0] * scale_percent / 100)
        h, w, c =  output_image.shape
        dim = (w, h)
        output_image = cv2.resize(output_image, dim, interpolation = cv2.INTER_AREA)
        if not os.path.exists(f'{output_dir}'):
            os.makedirs(f'{output_dir}')
        cv2.imwrite(f'{output_dir}/{i:03d}.png', output_image)
        frames.append(output_image)
        i += 1
    out = cv2.VideoWriter(f'{output_dir}/bev.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5, dim)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
    
def main():
    root_dir = 'outputs/2023-02-28/10-44-34/visualizations'
    assert os.path.exists(root_dir)
    print("Creating Gif.... Please Wait")
    create_gif(
        f'{root_dir}/input/camera5', 
        f'{root_dir}/object_detection/camera5', 
        f'{root_dir}/position_estimation/camera5',
        f'{root_dir}/zz_output/camera5'
    )

if __name__=="__main__":
    main()