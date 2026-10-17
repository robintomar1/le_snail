#!/usr/bin/env python3
"""Turn exported URDF joint-target sequences (scripts/export_hybrid_targets.py) into an actuator_log.py motion.

Maps target_rad through a joint map (configs/joint-maps/*.json) to goal ticks for the servo joints the map lists
(the gripper is skipped when the map has no entry for it), keeps the first --seconds, and writes the offsets about
the first sample. Prints the --start-ticks to pass to actuator_log.py, which ramps there from the present pose.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from le_snail.actuator_id import TICKS_PER_RAD, urdf_to_ticks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('targets', type=Path, help='episode .npz with t_s and target_rad')
    parser.add_argument('--joint-map', type=Path, required=True)
    parser.add_argument('--summary', type=Path, help='summary.json with joint_names (default: next to the targets)')
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    episode = np.load(args.targets)
    names = json.loads((args.summary or args.targets.with_name('summary.json')).read_text())['joint_names']
    joints = json.loads(args.joint_map.read_text())['joints']
    used = [name for name in names if name in joints]
    dt = float(np.median(np.diff(episode['t_s'])))
    n = round(args.seconds / dt)
    targets = episode['target_rad'][:n, [names.index(name) for name in used]]
    ticks = urdf_to_ticks(targets, [joints[name]['sign'] for name in used], [joints[name]['offset_ticks'] for name in used])
    ids = [joints[name]['id'] for name in used]
    start = ticks[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, offsets_ticks=ticks - start, rate_hz=1 / dt, ids=np.array(ids), name=args.out.stem,
             start_ticks=start, source=str(args.targets), joint_map=str(args.joint_map))
    print(f'{args.out}: {len(ticks)} samples at {1 / dt:.0f} Hz, ids {ids} ({", ".join(used)})')
    for name, lo, hi in zip(used, ticks.min(0), ticks.max(0)):
        print(f'  {name}: ticks {lo}..{hi}')
    print('--start-ticks ' + ' '.join(map(str, start)))


if __name__ == '__main__':
    main()
