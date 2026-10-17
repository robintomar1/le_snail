#!/usr/bin/env python3
"""Stage 0 of #12: dump every STS3215 register on the arm to JSON, read-only.

Sends only PING and READ, so nothing moves or changes. For each servo the dump holds the raw bytes,
decoded register values, any READ range that got no reply, and the registers LeRobot configure()
would change on connect at the pinned commit. Identification must happen in the post-configure
(deployment) state, so a non-empty difference means a configure() write is needed first; that is a
servo write and needs the operator's OK.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from le_snail.servo_bus import BAUDRATE, DUMP_RANGES, LEROBOT_COMMIT, STS3215_REGISTERS, ServoBus, dump_servos, open_bus_port


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('port')
    parser.add_argument('--ids', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--out', type=Path, default=Path('runs/actuator-id/stage0'))
    parser.add_argument('--adapter', default='ESP32-S3 CH343 pass-through to the Waveshare bus servo board')
    parser.add_argument('--timeout-ms', type=float, default=20.0)
    args = parser.parse_args()

    port = open_bus_port(args.port)
    try:
        servos = dump_servos(ServoBus(port, timeout_s=args.timeout_ms / 1000), args.ids)
    finally:
        port.close()
    utc = datetime.now(timezone.utc)
    record = {
        'schema': 'le-snail-servo-dump-1',
        'utc': utc.isoformat(),
        'port': args.port,
        'baudrate': BAUDRATE,
        'adapter': args.adapter,
        'register_table': f'huggingface/lerobot@{LEROBOT_COMMIT} STS_SMS_SERIES_CONTROL_TABLE',
        'dump_ranges': [list(r) for r in DUMP_RANGES],
        'register_addresses': {name: list(spec) for name, spec in STS3215_REGISTERS.items()},
        'servos': servos,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f'registers-{utc.strftime("%Y%m%dT%H%M%SZ")}.json'
    path.write_text(json.dumps(record, indent=1) + '\n')

    for servo_id, servo in servos.items():
        if not servo['answered']:
            print(f'id {servo_id}: no reply')
            continue
        r = servo['registers']
        print(f'id {servo_id}: model {r.get("Model_Number")} fw {r.get("Firmware_Major_Version")}.{r.get("Firmware_Minor_Version")} '
              f'P/I/D {r.get("P_Coefficient")}/{r.get("I_Coefficient")}/{r.get("D_Coefficient")} '
              f'acc {r.get("Acceleration")} max_acc {r.get("Maximum_Acceleration")} delay {r.get("Return_Delay_Time")} '
              f'mode {r.get("Operating_Mode")} torque {r.get("Torque_Enable")} pos {r.get("Present_Position")} '
              f'{r.get("Present_Voltage", 0) / 10:.1f} V {r.get("Present_Temperature")} C '
              f'failed {servo["failed_ranges"]} configure would change {servo["lerobot_configure_would_change"]}')
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
