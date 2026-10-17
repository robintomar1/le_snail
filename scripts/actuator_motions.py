#!/usr/bin/env python3
"""Write Stage 1 excitation motions for scripts/actuator_log.py: one joint at a time, the others holding.

Each file holds goal offsets in encoder ticks relative to the pose at start, beginning with a hold at
zero offset. actuator_log.py rejects any goal outside the servos' calibrated limits minus a margin.
Review the printed summary with the operator before the first powered run.
"""
import argparse
from pathlib import Path

import numpy as np

from le_snail.actuator_id import TICKS_PER_RAD, chirp, hold_then, steps


def ticks(degrees):
    return int(round(np.radians(degrees) * TICKS_PER_RAD))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ids', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--excite', type=int, nargs='+', help='ids to excite (default: all of --ids)')
    parser.add_argument('--rate-hz', type=float, default=50.0, help='command rate (the 50 Hz policy contract)')
    parser.add_argument('--step-deg', type=float, nargs='+', default=[2, 5, 10, 20, 30])
    parser.add_argument('--hold-s', type=float, default=1.0)
    parser.add_argument('--chirp-deg', type=float, default=5.0)
    parser.add_argument('--chirp-hz', type=float, nargs=2, default=[0.1, 5.0])
    parser.add_argument('--chirp-s', type=float, default=20.0)
    parser.add_argument('--out', type=Path, default=Path('runs/actuator-id/motions'))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    zero = np.zeros(len(args.ids), dtype=int)
    for servo_id in args.excite or args.ids:
        joint = args.ids.index(servo_id)
        amplitudes = [sign * ticks(deg) for deg in args.step_deg for sign in (1, -1)]
        motions = {
            f'steps-id{servo_id}': steps(zero, joint, amplitudes, args.hold_s, args.rate_hz),
            f'chirp-id{servo_id}': chirp(zero, joint, ticks(args.chirp_deg), *args.chirp_hz, args.chirp_s, args.rate_hz),
        }
        for name, offsets in motions.items():
            offsets = hold_then(np.vstack([zero, offsets]), args.hold_s, args.rate_hz)
            path = args.out / f'{name}.npz'
            np.savez(path, offsets_ticks=offsets, rate_hz=args.rate_hz, ids=np.array(args.ids), name=name)
            print(f'{path}: {len(offsets) / args.rate_hz:.1f} s, id {servo_id} offsets '
                  f'{offsets[:, joint].min()}..{offsets[:, joint].max()} ticks '
                  f'({np.degrees(offsets[:, joint].min() / TICKS_PER_RAD):.1f}..{np.degrees(offsets[:, joint].max() / TICKS_PER_RAD):.1f} deg)')


if __name__ == '__main__':
    main()
