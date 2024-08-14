import cv2
import numpy as np
from datasets.gradslam_datasets import load_dataset_config

def compute_K_optimal(K, distortion, distortion_model, image):
    wh = image.shape[:2][::-1]
    if distortion_model is None or distortion_model == "radtan":
        K_optimal, roi = cv2.getOptimalNewCameraMatrix(K, distortion, wh, alpha=0.0, newImgSize=wh)
        undistorted_image = cv2.undistort(image, K, distortion, None, K_optimal)
    elif distortion_model == "equidistant":
        K_optimal = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion, wh, R=np.eye(3), balance=0.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), K_optimal, wh, cv2.CV_16SC2)
        undistorted_image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    else:
        raise ValueError(f"Unknown distortion model {distortion_model}")

    cv2.imshow("original", image)
    cv2.imshow("undistorted", undistorted_image)
    cv2.waitKey(0)
    return K_optimal


if __name__ == "__main__":
    modality = 'visual' # thermal or visual
    gradslam_data_cfg = load_dataset_config(f'./configs/data/rrxio_{modality}.yaml')
    datasetdir = '/media/pi/BackupPlus/jhuai/data/rrxio/irs_rtvi_datasets_2021/mocap_easy'
    visual_fn = 'visual/1614701459.518784046.png'
    thermal_fn = 'thermal/1614701459.570277214.png'
    visual_path = f"{datasetdir}/{visual_fn}"
    thermal_path = f"{datasetdir}/{thermal_fn}"
    if modality == 'thermal':
        image = cv2.imread(thermal_path, cv2.IMREAD_UNCHANGED)
    else:
        image = cv2.imread(visual_path, cv2.IMREAD_UNCHANGED)

    K = np.eye(3)
    K[0, 0] = gradslam_data_cfg["camera_params"]["fx"]
    K[1, 1] = gradslam_data_cfg["camera_params"]["fy"]
    K[0, 2] = gradslam_data_cfg["camera_params"]["cx"]
    K[1, 2] = gradslam_data_cfg["camera_params"]["cy"]
    distortion = np.array(gradslam_data_cfg["camera_params"]["distortion"])
    distortion_model = gradslam_data_cfg["camera_params"]["distortion_model"]

    K_optimal = compute_K_optimal(K, distortion, distortion_model, image)
    print('The optimal K to undistort the {} image is\n{}'.format(modality, K_optimal))