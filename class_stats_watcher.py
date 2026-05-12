#!/usr/bin/env python3
"""Watch STCOcc result dir and print class distribution as each frame appears.

Usage:
    python class_stats_watcher.py                       # watch loop, default dir
    python class_stats_watcher.py --once                # process last frame and exit
    python class_stats_watcher.py --dir /some/other/dir
    python class_stats_watcher.py --top 8               # only show top-N classes

Notes:
    - Does NOT delete files; result_publisher will still consume them as usual.
    - Races against publisher's ~50ms unlink; expect to miss some frames when
      both are running. Stop the publisher if you need every frame.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np

OCC3D_NAMES = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation', 'free',
]

OPENOCC_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain',
    'manmade', 'vegetation', 'free',
]

SCHEMAS = {'occ3d': OCC3D_NAMES, 'openocc': OPENOCC_NAMES}

# RGB triples matching CLASS_COLORS_* in stcocc_result_publisher_node.py.
OCC3D_COLORS = [
    (  0,   0,   0), (255, 120,  50), (255, 192, 203), (255, 255,   0),
    (  0, 150, 245), (  0, 255, 255), (255, 127,   0), (255,   0,   0),
    (255, 240, 150), (135,  60,   0), (160,  32, 240), (255,   0, 255),
    (139, 137, 137), ( 75,   0,  75), (150, 240,  80), (230, 230, 250),
    (  0, 175,   0), (  0,   0,   0),
]

OPENOCC_COLORS = [
    (  0, 150, 245), (160,  32, 240), (135,  60,   0), (255, 255,   0),
    (  0, 255, 255), (255, 192, 203), (255, 127,   0), (255,   0,   0),
    (255, 240, 150), (255, 120,  50), (255,   0, 255), (139, 137, 137),
    ( 75,   0,  75), (150, 240,  80), (230, 230, 250), (  0, 175,   0),
    (  0,   0,   0),
]

COLOR_SCHEMAS = {'occ3d': OCC3D_COLORS, 'openocc': OPENOCC_COLORS}

RESET = '\x1b[0m'


def _ansi_fg(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    # Black (0,0,0) would be invisible on a dark terminal; nudge it to grey.
    if r == 0 and g == 0 and b == 0:
        r = g = b = 110
    return f'\x1b[38;2;{r};{g};{b}m'


def _supports_color() -> bool:
    if os.environ.get('NO_COLOR'):
        return False
    if not sys.stdout.isatty():
        return False
    term = os.environ.get('TERM', '')
    return term not in ('', 'dumb')


def stats_one(path: pathlib.Path):
    """Return (total_voxels, {class_id: count}) or None on read failure."""
    try:
        d = np.load(str(path))
        s = d['semantics']
    except (FileNotFoundError, OSError, KeyError, EOFError, ValueError):
        return None
    ids, cnts = np.unique(s, return_counts=True)
    return int(s.size), dict(zip(ids.tolist(), cnts.tolist()))


def print_histogram(name: str, total: int, hist: dict, top: int,
                    names: list, colors: list, use_color: bool) -> None:
    items = sorted(hist.items(), key=lambda x: -x[1])
    if top > 0:
        items = items[:top]
    print(f'\n[{name}] total {total} voxels')
    for cid, cnt in items:
        cname = names[cid] if 0 <= cid < len(names) else '???'
        pct = 100.0 * cnt / total
        bar = '#' * min(int(pct), 60)
        line = f'  class {cid:2d} {cname:22s}: {cnt:8d}  {pct:5.1f}%  {bar}'
        if use_color and 0 <= cid < len(colors):
            print(f'{_ansi_fg(colors[cid])}{line}{RESET}')
        else:
            print(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dir',
                   default='/home/d300/Desktop/STCOcc/ros_output',
                   help='Directory daemon writes frame_*.npz to')
    p.add_argument('--interval', type=float, default=0.05,
                   help='Poll interval (s)')
    p.add_argument('--top', type=int, default=0,
                   help='Only show top-N classes (0 = all non-zero)')
    p.add_argument('--once', action='store_true',
                   help='Process the latest existing frame and exit')
    p.add_argument('--schema', choices=list(SCHEMAS.keys()), default='openocc',
                   help='Class name table to use for labels')
    p.add_argument('--no-color', action='store_true',
                   help='Disable ANSI colored output')
    args = p.parse_args()

    names = SCHEMAS[args.schema]
    colors = COLOR_SCHEMAS[args.schema]
    use_color = (not args.no_color) and _supports_color()
    print(f'Using {args.schema} class names (color={"on" if use_color else "off"})')

    base = pathlib.Path(args.dir)
    if not base.exists():
        print(f'No such dir: {base}')
        return

    if args.once:
        files = sorted(base.glob('frame_*.npz'))
        if not files:
            print('No frames yet — start daemon first.')
            return
        res = stats_one(files[-1])
        if res is None:
            print(f'Failed to read {files[-1].name}')
            return
        print_histogram(files[-1].name, res[0], res[1], args.top, names, colors, use_color)
        return

    print(f'Watching {base}/  (Ctrl-C to stop)')
    seen: set[str] = set()
    try:
        while True:
            for f in sorted(base.glob('frame_*.npz')):
                if f.name in seen:
                    continue
                seen.add(f.name)
                res = stats_one(f)
                if res is None:
                    continue
                print_histogram(f.name, res[0], res[1], args.top, names, colors, use_color)

            if len(seen) > 1000:
                seen = set(list(seen)[-500:])
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
