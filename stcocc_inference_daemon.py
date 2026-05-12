#!/usr/bin/env python3
"""STCOcc inference daemon for ROS bag playback.

Watches /dev/shm/stcocc_input/ for frame_*.npz written by stcocc_collector_node,
runs STCOcc inference, and writes occupancy frame_*.npz back to
/home/d300/Desktop/STCOcc/ros_output/ for the result publisher.

Run natively (in stcocc env), NOT in Docker:
    conda activate stcocc
    cd /home/d300/Desktop/STCOcc
    python stcocc_inference_daemon.py \
        --config config/stcocc/stcocc_r50_704x256_16f_occ3d_36e.py \
        --weights /path/to/stcocc_occ3d.pth

Input .npz schema (written by collector):
    imgs                 (T=3, N=6, 3, 256, 704) float32 — normalized RGB
    sensor2ego           (T,    N, 4, 4)          float32
    ego2global           (T,    N, 4, 4)          float32
    cam2img              (N, 4, 4)                float32 — static intrinsics
    post_aug             (N, 4, 4)                float32 — identity
    bda                  (4, 4)                   float32 — identity
    curr_to_prev_ego_rt  (4, 4)                   float32
    sequence_group_idx   int
    start_of_sequence    bool
    frame_ts             float64

Output .npz:
    semantics  (200, 200, 16) uint8
    flow       (200, 200, 16, 2) float16
    frame_ts   float64
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import pathlib
import signal
import sys
import time
import traceback

import numpy as np
import torch

_DEFAULT_INPUT_DIR  = pathlib.Path('/dev/shm/stcocc_input')
_DEFAULT_OUTPUT_DIR = pathlib.Path('/home/d300/Desktop/STCOcc/ros_output')

TARGET_H, TARGET_W = 256, 704
NUM_CAMS = 6
NUM_FRAMES = 3   # curr + 2 adjacent (stereo extra_ref)


def _ego2global_to_can_bus(ego2global_4x4: np.ndarray) -> np.ndarray:
    """Reproduce nuScenes 18-d can_bus vector from current ego→world matrix.

    Layout (matches BEVFormer / STCOcc training pipeline):
      [0:3]   global xyz
      [3:7]   rotation quaternion (w, x, y, z)
      [7:16]  accel(3) + rot_rate(3) + velocity(3) — zeros (unused at inference)
      [16]    patch_angle in radians
      [17]    patch_angle in degrees, normalized to [0, 360)
    The detector overwrites [:3] and [-1] with per-frame deltas internally.
    """
    M = np.asarray(ego2global_4x4, dtype=np.float64)
    tx, ty, tz = M[:3, 3]
    R = M[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s; qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s; qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s; qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s; qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s; qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s; qz = 0.25 * s

    yaw_deg = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    if yaw_deg < 0.0:
        yaw_deg += 360.0
    yaw_rad = np.radians(yaw_deg)

    can_bus = np.zeros(18, dtype=np.float32)
    can_bus[0] = tx; can_bus[1] = ty; can_bus[2] = tz
    can_bus[3] = qw; can_bus[4] = qx; can_bus[5] = qy; can_bus[6] = qz
    can_bus[16] = yaw_rad
    can_bus[17] = yaw_deg
    return can_bus


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(config_path: str, weights_path: str):
    from mmcv import Config
    from mmdet3d.models import build_model
    from mmcv.runner import load_checkpoint

    cfg = Config.fromfile(config_path)
    cfg.model.train_cfg = None
    if hasattr(cfg.model, 'save_results'):
        cfg.model.save_results = False

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, weights_path, map_location='cpu', strict=False,
                    logger=logging.getLogger(__name__))
    model.cuda().eval()
    logging.info('Model loaded from %s', weights_path)
    return model


# ---------------------------------------------------------------------------
# Single-frame inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, npz_path: pathlib.Path, force_start_of_seq: bool = False) -> dict:
    t_load_start = time.perf_counter()
    data = np.load(str(npz_path), allow_pickle=False)
    t_load_end = time.perf_counter()

    imgs_np         = data['imgs']                # (T, N, 3, H, W)
    sensor2ego_np   = data['sensor2ego']          # (T, N, 4, 4)
    ego2global_np   = data['ego2global']          # (T, N, 4, 4)
    cam2img_np      = data['cam2img']             # (N, 4, 4)
    post_aug_np     = data['post_aug']            # (N, 4, 4)
    bda_np          = data['bda']                 # (4, 4)
    curr_to_prev_np = data['curr_to_prev_ego_rt'] # (4, 4)
    seq_idx         = int(data['sequence_group_idx'])
    start_of_seq    = bool(data['start_of_sequence']) or force_start_of_seq
    frame_ts        = float(data['frame_ts'])
    # ego→world at the current frame (T=0; broadcast across cams)
    ego2global_curr = ego2global_np[0, 0]

    # lidar2img / ego2lidar for backward_projection.point_sampling.
    # Reference frame "lidar" == ego (base_link); no BEV/img augmentation.
    #   ego2lidar = I
    #   lidar2img[i] = cam2img[i] @ inv(sensor2ego_curr[i])
    sensor2ego_curr = sensor2ego_np[0].astype(np.float64)            # (N, 4, 4) at T=0
    inv_s2e         = np.linalg.inv(sensor2ego_curr)
    lidar2img_curr  = (cam2img_np.astype(np.float64) @ inv_s2e).astype(np.float32)  # (N,4,4)

    T, N = imgs_np.shape[:2]
    assert T == NUM_FRAMES and N == NUM_CAMS, f'Bad input shape: T={T}, N={N}'

    # imgs: collector layout is (T, N, 3, H, W). Model's prepare_inputs does
    # imgs.view(B, N, num_frame, C, H, W), which expects N-outer, T-inner.
    # So transpose to (N, T, 3, H, W) then flatten to (1, N*T, 3, H, W).
    imgs_ord = imgs_np.transpose(1, 0, 2, 3, 4).reshape(1, N * T, 3, TARGET_H, TARGET_W)
    img_tensor = torch.from_numpy(imgs_ord).float().cuda()

    # sensor2ego / ego2global: model views as (B, T, N, 4, 4) — T-outer, N-inner.
    # Collector already stores (T, N, ...), so just flatten.
    s2e_t = torch.from_numpy(sensor2ego_np.reshape(1, T * N, 4, 4)).float().cuda()
    e2g_t = torch.from_numpy(ego2global_np.reshape(1, T * N, 4, 4)).float().cuda()

    # cam2img / post_aug: static; tile to (1, T*N, 4, 4) with T-outer order.
    c2i_t      = torch.from_numpy(np.tile(cam2img_np[None], (T, 1, 1, 1))
                                   .reshape(1, T * N, 4, 4)).float().cuda()
    post_aug_t = torch.from_numpy(np.tile(post_aug_np[None], (T, 1, 1, 1))
                                   .reshape(1, T * N, 4, 4)).float().cuda()

    bda_t = torch.from_numpy(bda_np[None]).float().cuda()        # (1, 4, 4)

    img_inputs = [(img_tensor, s2e_t, e2g_t, c2i_t, post_aug_t, bda_t)]

    img_metas = [{
        'sequence_group_idx': seq_idx,
        'start_of_sequence':  start_of_seq,
        'curr_to_prev_ego_rt': curr_to_prev_np.astype(np.float32),
        'sample_idx': npz_path.stem,
        'scene_name': 'ros_bag',
        'index': 0,
        'can_bus': _ego2global_to_can_bus(ego2global_curr),
        'lidar2img': torch.from_numpy(lidar2img_curr),
        'ego2lidar': torch.eye(4, dtype=torch.float32),
    }]

    t_forward_start = time.perf_counter()
    results = model.simple_test(
        points=[None],
        img_metas=img_metas,
        img_inputs=img_inputs,
    )
    torch.cuda.synchronize()
    t_forward_end = time.perf_counter()

    # results[0] is a dict with 'occ_results' (B, W, H, Z) and 'flow_results'
    occ  = results[0]['occ_results'][0]   # (W, H, Z) uint8 → 200×200×16
    flow = results[0]['flow_results'][0]  # (W, H, Z, 2) float16

    return {
        'semantics': occ,
        'flow':      flow,
        'frame_ts':  frame_ts,
        'timing_ms': {
            'load':    round((t_load_end - t_load_start) * 1000, 2),
            'forward': round((t_forward_end - t_forward_start) * 1000, 2),
        },
    }


# ---------------------------------------------------------------------------
# Main watch loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
        default='config/stcocc/stcocc_r50_704x256_16f_occ3d_36e.py')
    parser.add_argument('--weights', required=True,
        help='Path to STCOcc .pth checkpoint')
    parser.add_argument('--poll_interval', type=float, default=0.05)
    parser.add_argument('--input_dir',  type=pathlib.Path, default=_DEFAULT_INPUT_DIR)
    parser.add_argument('--output_dir', type=pathlib.Path, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # STCOcc uses mmdet3d.models registry — import the project to register modules
    project_root = pathlib.Path(__file__).resolve().parent
    os.chdir(str(project_root))
    sys.path.insert(0, str(project_root))
    # Trigger module registration
    importlib.import_module('mmdet3d.models.stcocc')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.input_dir.mkdir(parents=True, exist_ok=True)

    logging.info('Loading model...')
    model = load_model(args.config, args.weights)
    logging.info('Watching %s', args.input_dir)

    processed: set[str] = set()
    forward_times_ms: list[float] = []
    first_frame_done = False

    def _shutdown(*_):
        if forward_times_ms:
            n = len(forward_times_ms)
            avg = sum(forward_times_ms) / n
            logging.info('Shutdown: %d frames, avg forward %.1f ms', n, avg)
        raise SystemExit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        pending = sorted(args.input_dir.glob('frame_*.npz'))
        if pending:
            latest = pending[-1]
            stale = [p for p in pending[:-1] if p.name not in processed]
            if stale:
                logging.info('Dropping %d stale frame(s); keeping %s',
                             len(stale), latest.name)
            for p in stale:
                processed.add(p.name)
                try:
                    p.unlink()
                except OSError:
                    pass

            if latest.name not in processed:
                try:
                    result = run_inference(model, latest,
                                           force_start_of_seq=not first_frame_done)
                    first_frame_done = True
                    out_path = args.output_dir / latest.name
                    tmp_path = args.output_dir / (latest.name + '.tmp')
                    with open(tmp_path, 'wb') as f:
                        np.savez(f,
                                 semantics=result['semantics'],
                                 flow=result['flow'],
                                 frame_ts=np.float64(result['frame_ts']))
                    tmp_path.rename(out_path)
                    fwd = result['timing_ms']['forward']
                    forward_times_ms.append(fwd)
                    avg = sum(forward_times_ms) / len(forward_times_ms)
                    logging.info(
                        'frame %s | forward %.1f ms | avg %.1f ms (n=%d)',
                        latest.stem, fwd, avg, len(forward_times_ms),
                    )
                except Exception:
                    logging.error('Error on %s:\n%s', latest.name, traceback.format_exc())
                finally:
                    processed.add(latest.name)
                    try:
                        latest.unlink()
                    except OSError:
                        pass

        if len(processed) > 500:
            processed = set(list(processed)[-200:])
        time.sleep(args.poll_interval)


if __name__ == '__main__':
    main()
