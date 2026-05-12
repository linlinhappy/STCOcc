#!/usr/bin/env python3
"""Watch STCOcc result dir and print flow magnitude distribution per frame.

Helps tune flow_min_speed / flow_arrow_scale in params_*.yaml.

Usage:
    python flow_stats_watcher.py                       # watch loop
    python flow_stats_watcher.py --once                # one-shot on latest frame
    python flow_stats_watcher.py --by-class            # break down per class
    python flow_stats_watcher.py --threshold 0.5       # change "non-zero" cutoff

Notes:
    - Reads d['flow'] of shape (Gx, Gy, Gz, 2) in model units (~m/frame).
    - Races against result_publisher's ~50ms unlink; expect some misses.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

OPENOCC_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation', 'free',
]
OCC3D_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation', 'free',
]
SCHEMAS = {'occ3d': OCC3D_NAMES, 'openocc': OPENOCC_NAMES}


def load_frame(path: pathlib.Path):
    try:
        d = np.load(str(path))
        flow = d['flow'].astype(np.float32)
        sem = d['semantics']
    except (FileNotFoundError, OSError, KeyError, EOFError, ValueError):
        return None
    return sem, flow


def overall_stats(mag: np.ndarray, threshold: float) -> dict:
    nz = mag[mag > threshold]
    total = mag.size
    if len(nz) == 0:
        return {'total': total, 'nz': 0}
    return {
        'total': total,
        'nz':    len(nz),
        'min':   float(nz.min()),
        'max':   float(nz.max()),
        'mean':  float(nz.mean()),
        'med':   float(np.median(nz)),
        'p50':   float(np.percentile(nz, 50)),
        'p75':   float(np.percentile(nz, 75)),
        'p90':   float(np.percentile(nz, 90)),
        'p99':   float(np.percentile(nz, 99)),
    }


def print_overall(name: str, s: dict, threshold: float) -> None:
    pct = 100.0 * s['nz'] / s['total'] if s['total'] else 0.0
    print(f'\n[{name}]')
    print(f'  voxels with |flow| > {threshold}: {s["nz"]:>8} / {s["total"]}  ({pct:.2f}%)')
    if s['nz'] == 0:
        return
    print(f'  min={s["min"]:.2f}  max={s["max"]:.2f}  mean={s["mean"]:.2f}  median={s["med"]:.2f}')
    print(f'  percentiles  p50={s["p50"]:.2f}  p75={s["p75"]:.2f}  '
          f'p90={s["p90"]:.2f}  p99={s["p99"]:.2f}')
    # Recommendation
    suggested_min = max(s['p75'], threshold)
    suggested_scale = max(0.05, min(1.0, 2.0 / s['p99']))
    print(f'  → suggested flow_min_speed   ≈ {suggested_min:.2f}')
    print(f'  → suggested flow_arrow_scale ≈ {suggested_scale:.2f} '
          f'(targets ~2 m max arrow)')


def print_by_class(sem: np.ndarray, mag: np.ndarray, threshold: float,
                   names: list, top: int = 8) -> None:
    print('\n  per-class (only classes with any |flow| > threshold):')
    print(f'    {"class":<25s} {"nz_count":>8} {"median":>8} {"p90":>8} {"max":>8}')
    rows = []
    for cid in range(len(names)):
        m = mag[sem == cid]
        nz = m[m > threshold]
        if len(nz) == 0:
            continue
        rows.append((cid, names[cid], len(nz), float(np.median(nz)),
                     float(np.percentile(nz, 90)), float(nz.max())))
    rows.sort(key=lambda r: -r[2])  # by count desc
    for cid, cname, cnt, med, p90, mx in rows[:top]:
        print(f'    {cid:2d} {cname:<22s} {cnt:>8} {med:>8.2f} {p90:>8.2f} {mx:>8.2f}')


def process(path: pathlib.Path, threshold: float, by_class: bool, names: list):
    res = load_frame(path)
    if res is None:
        return
    sem, flow = res
    mag = np.linalg.norm(flow, axis=-1)        # (Gx, Gy, Gz)
    stats = overall_stats(mag, threshold)
    print_overall(path.name, stats, threshold)
    if by_class and stats['nz'] > 0:
        print_by_class(sem, mag, threshold, names)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='/home/d300/Desktop/STCOcc/ros_output')
    p.add_argument('--interval', type=float, default=0.1)
    p.add_argument('--threshold', type=float, default=0.1,
                   help='Magnitude cutoff for "non-zero" voxel counting (default 0.1)')
    p.add_argument('--once', action='store_true')
    p.add_argument('--by-class', action='store_true',
                   help='Also break down flow stats per semantic class')
    p.add_argument('--schema', choices=list(SCHEMAS.keys()), default='openocc')
    args = p.parse_args()

    names = SCHEMAS[args.schema]
    base = pathlib.Path(args.dir)
    if not base.exists():
        print(f'No such dir: {base}')
        return

    if args.once:
        files = sorted(base.glob('frame_*.npz'))
        if not files:
            print('No frames yet — start daemon first.')
            return
        process(files[-1], args.threshold, args.by_class, names)
        return

    print(f'Watching {base}/  (Ctrl-C to stop)')
    seen: set[str] = set()
    try:
        while True:
            for f in sorted(base.glob('frame_*.npz')):
                if f.name in seen:
                    continue
                seen.add(f.name)
                process(f, args.threshold, args.by_class, names)

            if len(seen) > 1000:
                seen = set(list(seen)[-500:])
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
