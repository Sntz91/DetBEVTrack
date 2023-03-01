from src.camera import GeneratedFrameCamera
import hydra

@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    camera = GeneratedFrameCamera(
        name=cfg.cameras.camera5.name, 
        homography_matrix_filename=cfg.cameras.camera5.files.homography_matrix,
        frame_dir=cfg.cameras.camera5.directories.frames,
        mask_dir=cfg.cameras.camera5.directories.masks,
        gt_filename=cfg.gt.positions,
        output_dir=cfg.cameras.camera5.directories.output,
        reference_image_filename=cfg.gt.reference_img
    )
    camera.run(skip=0)
    print('done')


if __name__=='__main__':
    main()