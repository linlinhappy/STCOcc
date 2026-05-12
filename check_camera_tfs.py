#!/usr/bin/env python3
"""Print base_link → <camera> TFs for STCOcc's 6 cameras.

Usage:
    ros2 run tf2_ros static_transform_publisher ... (must be running)
    python3 check_camera_tfs.py
"""
import math
import sys

import rclpy
from rclpy.node import Node
import tf2_ros


CAMS = [
    ('gige_100_fl_hdr', 'CAM_FRONT_LEFT'),
    ('gige_100_f_hdr',  'CAM_FRONT'),
    ('gige_100_fr_hdr', 'CAM_FRONT_RIGHT'),
    ('gige_60_bl',      'CAM_BACK_LEFT'),
    ('gige_100_b_hdr',  'CAM_BACK'),
    ('gige_60_br',      'CAM_BACK_RIGHT'),
]


def quat_to_rpy(qx, qy, qz, qw):
    """Quaternion (x,y,z,w) → (roll, pitch, yaw) in degrees."""
    sinr = 2 * (qw * qx + qy * qz)
    cosr = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def main():
    rclpy.init()
    node = Node('check_camera_tfs')
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, node)

    # Spin for a couple seconds to fill the buffer
    import time
    deadline = time.time() + 3.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)

    print(f'\n{"camera":<20} {"role":<18} {"x(前+/後-)":>12} {"y(左+/右-)":>12} {"z(高)":>8}  '
          f'{"roll":>7} {"pitch":>7} {"yaw":>7}')
    print('-' * 100)

    for cam, role in CAMS:
        try:
            tr = buf.lookup_transform('base_link', cam, rclpy.time.Time(),
                                      timeout=rclpy.duration.Duration(seconds=0.5))
            t = tr.transform.translation
            r = tr.transform.rotation
            roll, pitch, yaw = quat_to_rpy(r.x, r.y, r.z, r.w)
            print(f'{cam:<20} {role:<18} {t.x:>12.3f} {t.y:>12.3f} {t.z:>8.3f}  '
                  f'{roll:>7.1f} {pitch:>7.1f} {yaw:>7.1f}')
        except Exception as e:
            print(f'{cam:<20} {role:<18}    ❌ NOT FOUND: {e}')

    print('\nExpected sane ranges:')
    print('  x: -1.0 ~ +3.0 m   (CAM_FRONT 前方最遠，CAM_BACK 後方最遠)')
    print('  y: -1.0 ~ +1.0 m   (側面相機 ~±0.9，前後置中相機 ~0)')
    print('  z:  1.0 ~ +2.0 m   (車頂相機 ~1.5-1.8，較低位置 ~1.2-1.4)')
    print('  pitch 接近 0 或微微下傾 (-15° ~ +5°)')
    print('  yaw: F=0°, FL=+45°, FR=-45°, B=180°, BL=+135°, BR=-135°')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
