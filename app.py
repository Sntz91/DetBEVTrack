from src.camera import GeneratedFrameCamera
import hydra

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    camera = GeneratedFrameCamera(
        name=cfg.cameras.cameravup.name, 
        homography_matrix_filename=cfg.cameras.cameravup.files.homography_matrix,
        frame_dir=cfg.cameras.cameravup.directories.frames,
        mask_dir=cfg.cameras.cameravup.directories.masks,
        gt_filename=cfg.gt.positions,
        output_dir=cfg.cameras.cameravup.directories.output,
        reference_image_filename=cfg.gt.reference_img,
        fake_detections_dir=cfg.fake_detector.fake_detections_dir
    )
    camera.run(skip=0)
    print('done')


if __name__=='__main__':
    main()