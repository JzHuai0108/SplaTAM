import os


import cv2
from cv_bridge import CvBridge

import imageio
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2


def pointcloud2_msg_to_pcl(msg):
    """
    Converts a pointcloud2 message to a pcl pointcloud.

    Args:
    - msg (PointCloud2): The pointcloud2 message to be converted.
        struct RadarPointCloudType
        {
          PCL_ADD_POINT4D;      // position in [m]
          float snr_db;         // CFAR cell to side noise ratio in [dB]
          float v_doppler_mps;  // Doppler velocity in [m/s]
          float noise_db;       // CFAR noise level of the side of the detected cell in [dB]
          float range;          // range in [m]
          EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        };
    Returns:
    - pcl (np.ndarray): The pointcloud as a numpy array.
    """


    points = pc2.read_points(msg, field_names=("x", "y", "z", "snr_db", "v_doppler_mps", "noise_db", "range"), skip_nans=True)
    pcl = np.array(list(points))
    return pcl

def transform_points_to_camera_frame(points, T, intrinsics, scale=500):
    """
    Transforms points from the radar frame to the camera frame and projects them onto the camera image.

    Args:
    - points (np.ndarray): The pointcloud as a numpy array.
    - T (np.ndarray): The transformation matrix from the radar frame to the camera frame.
    - intrinsics (np.ndarray): The camera intrinsics matrix, w, h, fx, fy, cx, cy, and others.
    - scale (float): The scale factor for the depth image.
    Returns:
    - depth (np.ndarray): The depth image.
    """
    depth = np.zeros((intrinsics[1], intrinsics[0]), dtype=np.int32)
    fx, fy, cx, cy = intrinsics[2:6]
    count = 0
    for i, pall in enumerate(points):
        p = np.append(pall[:3], 1)
        p = np.matmul(T, p)
        u = p[0] * fx / p[2] + cx
        v = p[1] * fy / p[2] + cy
        ru = round(u)
        rv = round(v)
        max_depth = 65535
        if 0 <= ru < intrinsics[0] and 0 <= rv < intrinsics[1]:
            val = p[2] * scale
            if val > max_depth:
                print('Warning: depth value {} exceeds {}.'.format(val, max_depth))
                val = max_depth
            depth[rv, ru] = int(val)
        else:
            count += 1
    return depth, count


def rrxio_bag_to_folder(bag_path, output_folder):
    """
    Extracts visual, thermal, and radar data from an RRXIO bag file and saves them to a folder.

    Args:
    - bag_path (str): The path to the RRXIO bag file.
    - output_folder (str): The path to the folder where the extracted data will be saved.
    - radar_to_thermal (bool): If True, the radar data will be transformed to the thermal frame. Otherwise,
        the radar data will be saved in the visual camera frame in a png format.
    """
    visual_topic = '/sensor_platform/camera_visual/img'
    thermal_topic = '/sensor_platform/camera_thermal/img'
    radar_topic = '/sensor_platform/radar/scan'
    radar_trigger_topic = '/sensor_platform/radar/trigger'

    visual_intrinsics = [640, 512, 372.51, 372.51, 318.77, 253.24,
                         0.013404032824313381, 0.013570186060580948, -0.011005228038287808, 0.0040591597486]

    thermal_intrinsics = [640, 512, 404.98, 405.06, 319.05, 251.84,
                          -0.09202092749042277, 0.04012198239889151, -0.03795923559427829, 0.010728597805678742]
    # transforms from the radar frame to the thermal camera frame: p_thermal = T_thermal_radar * p_radar
    visual_T_radar = np.array([[9.99978632e-01, 6.66223802e-03, -5.69217613e-03, -1.96677240e-02],
                                 [5.85607039e-04, -2.16606087e-02, 9.99748301e-01, 7.38610120e-02],
                                 [-6.55759302e-03, 9.99742953e-01, 2.16989813e-02, 2.08262180e-02],
                                 [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    # transforms from the radar frame to the thermal camera frame:  p_thermal = T_thermal_radar * p_radar
    thermal_T_radar = np.array([[9.99943971e-01, 1.06349530e-02, -4.27393669e-03, -1.90940752e-02],
                                [-6.79812685e-04, -2.79824291e-02, 9.99597968e-01, 3.46560369e-02],
                                [-1.05637311e-02, 9.99551842e-01, 2.80291899e-02, 3.09737853e-02],
                                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

    visual_dir = os.path.join(output_folder, 'visual')
    thermal_dir = os.path.join(output_folder, 'thermal')
    radarv_dir = os.path.join(output_folder, 'radarv')
    radart_dir = os.path.join(output_folder, 'radart')
    os.makedirs(visual_dir, exist_ok=True)
    os.makedirs(thermal_dir, exist_ok=True)
    os.makedirs(radarv_dir, exist_ok=True)
    os.makedirs(radart_dir, exist_ok=True)

    visual_frames = []
    thermal_frames = []
    radarv_frames = []
    radart_frames = []
    bag = rosbag.Bag(bag_path)
    bridge = CvBridge()
    for topic, msg, t in bag.read_messages():
        if topic == visual_topic:
            frame_time = msg.header.stamp
            basename = f'visual/{frame_time.secs}.{frame_time.nsecs:09d}.png'
            frame_path = os.path.join(output_folder, basename)
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            visual_frames.append((frame_time, basename))
            cv2.imwrite(frame_path, cv_image)
        elif topic == thermal_topic:
            frame_time = msg.header.stamp
            basename = f'thermal/{frame_time.secs}.{frame_time.nsecs:09d}.png'
            frame_path = os.path.join(output_folder, basename)
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            thermal_frames.append((frame_time, basename))
            cv2.imwrite(frame_path, cv_image)
        elif topic == radar_topic:
            pointcloud = pointcloud2_msg_to_pcl(msg)
            frame_time = msg.header.stamp
            depthv, outlierv = transform_points_to_camera_frame(pointcloud, visual_T_radar, visual_intrinsics, scale = 500)
            basenamev = f'radarv/{frame_time.secs}.{frame_time.nsecs:09d}.png'
            radarv_frames.append((frame_time, basenamev))
            # if outlierv > 0.2 * len(pointcloud):
            #     print('radarv {} : {} out of {}'.format(basenamev, outlierv, len(pointcloud)))
            imageio.imwrite(os.path.join(output_folder, basenamev), depthv)

            deptht, outliert = transform_points_to_camera_frame(pointcloud, thermal_T_radar, thermal_intrinsics, scale = 500)
            basenamet = f'radart/{frame_time.secs}.{frame_time.nsecs:09d}.png'
            radart_frames.append((frame_time, basenamet))
            # if outliert > 0.2 * len(pointcloud):
            #     print('radart {} : {} out of {}'.format(basenamet, outliert, len(pointcloud)))
            # if outliert != outlierv:
            #     print('radart {} and radarv {} have different number of outliers: {} and {} out of {} points'.format(
            #         basenamet, basenamev, outliert, outlierv, len(pointcloud)))

            imageio.imwrite(os.path.join(output_folder, basenamet), deptht)

    bag.close()
    with open(os.path.join(output_folder, 'visual.txt'), 'w') as f:
        for frame_time, basename in visual_frames:
            f.write(f'{frame_time.secs}.{frame_time.nsecs:09d} {basename}\n')
    with open(os.path.join(output_folder, 'thermal.txt'), 'w') as f:
        for frame_time, basename in thermal_frames:
            f.write(f'{frame_time.secs}.{frame_time.nsecs:09d} {basename}\n')
    with open(os.path.join(output_folder, 'radarv.txt'), 'w') as f:
        for frame_time, basename in radarv_frames:
            f.write(f'{frame_time.secs}.{frame_time.nsecs:09d} {basename}\n')
    with open(os.path.join(output_folder, 'radart.txt'), 'w') as f:
        for frame_time, basename in radart_frames:
            f.write(f'{frame_time.secs}.{frame_time.nsecs:09d} {basename}\n')
    print('Saved {} visual frames, {} thermal frames, {} radarv frames, and {} radart frames to {}'.format(
        len(visual_frames), len(thermal_frames), len(radarv_frames), len(radart_frames), output_folder))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_path', type=str)
    parser.add_argument('output_folder', type=str)
    args = parser.parse_args()
    rrxio_bag_to_folder(args.bag_path, args.output_folder)
