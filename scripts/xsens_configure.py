#!/usr/bin/env python3
"""Set the Xsens MTi output configuration needed by the policy, with a saved backup and verification.

Without --apply this only reads the current configuration and prints the planned change.
The output configuration is stored on the device and survives power cycles.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct
import time

import numpy as np

from le_snail.hardware import (
    XDI_ACCELERATION, XDI_PACKET_COUNTER, XDI_QUATERNION, XDI_RATE_OF_TURN, XDI_SAMPLE_TIME_FINE,
    XMID_ERROR, XMID_GOTO_CONFIG, XMID_GOTO_MEASUREMENT, XMID_MTDATA2, XMID_REQ_OUTPUT_CONFIGURATION,
    ImuDecoder, XbusReader, parse_mtdata2, xbus_frame,
)

# SetOutputConfiguration shares its message id with the request; the payload is a list of
# (data identifier, frequency) pairs as big-endian uint16 (xspublic mtibasedevice.cpp / xsdevice_def.cpp).
XMID_SET_OUTPUT_CONFIGURATION = XMID_REQ_OUTPUT_CONFIGURATION
XDI_STATUS_WORD = 0xE020
EVERY_PACKET = 0xFFFF


def planned(rate_hz):
    return [
        (XDI_PACKET_COUNTER, EVERY_PACKET),
        (XDI_SAMPLE_TIME_FINE, EVERY_PACKET),
        (XDI_QUATERNION, rate_hz),        # float32, ENU (format bits 0)
        (XDI_RATE_OF_TURN, rate_hz),      # float32, rad/s
        (XDI_ACCELERATION, rate_hz),      # float32, m/s^2 (diagnostics)
        (XDI_STATUS_WORD, EVERY_PACKET),
    ]


def encode(config):
    return b''.join(struct.pack('>HH', identifier, hz) for identifier, hz in config)


def decode(payload):
    return [tuple(pair) for pair in struct.iter_unpack('>HH', payload[:len(payload) // 4 * 4])]


def describe(config):
    return [{'data_id': f'0x{identifier:04X}', 'hz': hz} for identifier, hz in config]


# XRV_DATAOVERFLOW (xspublic xsresultvalue.h): an asynchronous report that the device produced more
# data than the bus carried, e.g. while nothing was reading the port. It is not a reply to a command.
XRV_DATAOVERFLOW = 0x29
overflow_reports = []


def transact(port, reader, mid, payload=b'', timeout=1.5, attempts=3):
    for _ in range(attempts):
        port.reset_input_buffer()
        port.write(xbus_frame(mid, payload))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for reply_mid, reply in reader.feed(port.read(port.in_waiting or 1)):
                if reply_mid == XMID_ERROR:
                    if reply[:1] == bytes((XRV_DATAOVERFLOW,)):
                        overflow_reports.append(f'0x{mid:02X}')
                        continue
                    raise RuntimeError(f'Xsens error reply 0x{reply.hex()} to message 0x{mid:02X}')
                if reply_mid == mid + 1:
                    return reply
    raise RuntimeError(f'No acknowledgement for message 0x{mid:02X} after {attempts} attempts')


def verify_stream(port, seconds):
    reader, decoder = XbusReader(), ImuDecoder(np.eye(3))
    port.reset_input_buffer()
    decoded, ids, errors = [], None, set()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for mid, payload in reader.feed(port.read(port.in_waiting or 1)):
            if mid != XMID_MTDATA2:
                continue
            ids = sorted(f'0x{i:04X}' for i in parse_mtdata2(payload))
            try:
                decoded.append(decoder.decode(payload, time.monotonic()))
            except ValueError as exc:
                errors.add(str(exc))
    result = {'stream_data_ids': ids, 'packets_decoded': len(decoded), 'decode_errors': sorted(errors)}
    if len(decoded) > 1:
        result['rate_hz'] = round((len(decoded) - 1) / (decoded[-1]['sample_at'] - decoded[0]['sample_at']), 2)
        result['mean_sensor_gravity'] = np.mean([d['gravity'] for d in decoded], axis=0).round(4).tolist()
        result['mean_gyro_rad_s'] = np.mean([d['gyro'] for d in decoded], axis=0).round(5).tolist()
        result['gyro_std_rad_s'] = np.std([d['gyro'] for d in decoded], axis=0).round(5).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--baudrate', type=int, required=True)
    parser.add_argument('--rate-hz', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, required=True, help='where the backup and result are written')
    parser.add_argument('--restore', type=Path, help='restore the output configuration saved in this backup file')
    parser.add_argument('--apply', action='store_true', help='actually write the configuration to the device')
    parser.add_argument('--verify-seconds', type=float, default=3.0)
    args = parser.parse_args()
    if not 1 <= args.rate_hz <= 400:
        parser.error('--rate-hz must be in [1, 400]')
    target = ([tuple(int(v, 0) if isinstance(v, str) else v for v in (row['data_id'], row['hz']))
               for row in json.loads(args.restore.read_text())['before']]
              if args.restore else planned(args.rate_hz))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import serial
    port = serial.Serial(args.port, baudrate=args.baudrate, timeout=0.05)
    reader = XbusReader(max_consecutive_errors=10_000)
    report = {'time_utc': stamp, 'port': args.port, 'baudrate': args.baudrate, 'applied': False}
    try:
        transact(port, reader, XMID_GOTO_CONFIG)
        before = decode(transact(port, reader, XMID_REQ_OUTPUT_CONFIGURATION))
        report['before'] = describe(before)
        report['target'] = describe(target)
        backup = args.output_dir / f'output-config-backup-{stamp}.json'
        backup.write_text(json.dumps({'before': report['before']}, indent=2) + '\n')
        report['backup'] = str(backup)
        if sorted(before) == sorted(target):
            report['note'] = 'device already has the target configuration'
        elif args.apply:
            acked = decode(transact(port, reader, XMID_SET_OUTPUT_CONFIGURATION, encode(target), timeout=3.0))
            report['acknowledged'] = describe(acked)
            after = decode(transact(port, reader, XMID_REQ_OUTPUT_CONFIGURATION))
            report['after'] = describe(after)
            if sorted(after) != sorted(target):
                raise RuntimeError('Device configuration after write differs from the target; see report')
            report['applied'] = True
        else:
            report['note'] = 'dry run: re-run with --apply to write the target configuration'
    finally:
        try:
            transact(port, reader, XMID_GOTO_MEASUREMENT)
        finally:
            if report.get('applied') or report.get('note', '').startswith('device already'):
                report['verification'] = verify_stream(port, args.verify_seconds)
            port.close()
            report['overflow_reports_ignored'] = overflow_reports
            (args.output_dir / f'output-config-result-{stamp}.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
