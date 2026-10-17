#!/usr/bin/env python3
"""Read-only pose snapshot: SYNC_READ the state block for a few seconds, print median ticks and spread per servo.

With --track, also report the min/max reached per servo, so an operator can sweep a joint by hand (torque off)
to its stops during the window. Never writes to a servo. Saves runs/actuator-id/pose/<label>-<utc>.json.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from le_snail.actuator_id import decode_state
from le_snail.servo_bus import STATE_BLOCK, ServoBus, open_bus_port


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('port')
    parser.add_argument('label')
    parser.add_argument('--ids', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--seconds', type=float, default=1.0)
    parser.add_argument('--rate-hz', type=float, default=100.0)
    parser.add_argument('--track', action='store_true', help='report min/max per servo (for a hand sweep)')
    parser.add_argument('--out', default='runs/actuator-id/pose')
    args = parser.parse_args()

    bus = ServoBus(open_bus_port(args.port))
    rows, lost, period = [], 0, 1 / args.rate_hz
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        tick = time.monotonic()
        result = bus.sync_read(args.ids, *STATE_BLOCK)
        if result.lost:
            lost += 1
        else:
            rows.append([decode_state(result.replies[i][1])['position'] for i in args.ids])
        time.sleep(max(0.0, period - (time.monotonic() - tick)))
    pos = np.array(rows)
    summary = {
        'label': args.label, 'utc': datetime.now(timezone.utc).isoformat(), 'ids': args.ids,
        'samples': len(rows), 'lost': lost,
        'median_ticks': np.median(pos, 0).tolist(), 'spread_ticks': np.ptp(pos, 0).tolist(),
        'min_ticks': pos.min(0).tolist(), 'max_ticks': pos.max(0).tolist(),
    }
    if args.track:
        summary['positions'] = pos.tolist()
    print(f'{args.label}: {len(rows)} samples, {lost} lost')
    for k, i in enumerate(args.ids):
        extra = f'  min {pos[:, k].min()}  max {pos[:, k].max()}' if args.track else ''
        print(f'  id {i}: median {np.median(pos[:, k]):7.1f}  spread {np.ptp(pos[:, k]):4d}{extra}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{args.label}-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.json'
    path.write_text(json.dumps(summary, indent=1))
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
