#!/usr/bin/env python3
"""Report an Xsens MTi's identity and stored settings, then sample its measurement stream.

Nothing is written to the device's stored settings. Requests require configuration mode, so the
device is switched to configuration mode and returned to measurement mode before exiting.
"""
import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np

from le_snail.hardware import (
    XMID_ERROR, XMID_GOTO_CONFIG, XMID_GOTO_MEASUREMENT, XMID_MTDATA2, XMID_REQ_ALIGNMENT_ROTATION,
    XMID_REQ_AVAILABLE_FILTER_PROFILES, XMID_REQ_DID, XMID_REQ_FILTER_PROFILE, XMID_REQ_FIRMWARE_REVISION,
    XMID_REQ_OUTPUT_CONFIGURATION, XMID_REQ_PRODUCT_CODE, ImuDecoder, XbusReader, parse_mtdata2, xbus_frame,
)


def transact(port, reader, mid, payload=b'', timeout=1.0):
    port.write(xbus_frame(mid, payload))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for reply_mid, reply in reader.feed(port.read(port.in_waiting or 1)):
            if reply_mid == XMID_ERROR:
                return {'error': reply.hex()}
            if reply_mid == mid + 1:
                return reply
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True)
    parser.add_argument('--baudrate', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=3.0, help='measurement stream sample duration')
    parser.add_argument('--raw-dump', type=Path, help='save raw measurement bytes as a test fixture')
    args = parser.parse_args()
    import serial
    port = serial.Serial(args.port, baudrate=args.baudrate, timeout=0.05)
    reader = XbusReader(max_consecutive_errors=10_000)
    report = {'port': args.port, 'baudrate': args.baudrate}
    try:
        if transact(port, reader, XMID_GOTO_CONFIG) is None:
            raise SystemExit('No GoToConfig acknowledgement: check port, baud rate and power')
        def text(reply):
            return reply.decode('ascii', 'replace').strip('\x00 ') if isinstance(reply, bytes) else reply
        report['device_id'] = (lambda r: r.hex().upper() if isinstance(r, bytes) else r)(transact(port, reader, XMID_REQ_DID))
        report['product_code'] = text(transact(port, reader, XMID_REQ_PRODUCT_CODE))
        firmware = transact(port, reader, XMID_REQ_FIRMWARE_REVISION)
        report['firmware'] = '.'.join(map(str, firmware[:3])) if isinstance(firmware, bytes) and len(firmware) >= 3 else firmware
        output = transact(port, reader, XMID_REQ_OUTPUT_CONFIGURATION)
        if isinstance(output, bytes):
            report['output_configuration'] = [
                {'data_id': f'0x{i:04X}', 'hz': f} for i, f in struct.iter_unpack('>HH', output[:len(output)//4*4])]
        else:
            report['output_configuration'] = output
        for frame, name in ((0, 'sensor'), (1, 'local')):
            reply = transact(port, reader, XMID_REQ_ALIGNMENT_ROTATION, bytes((frame,)))
            report[f'alignment_{name}_q_wxyz'] = list(struct.unpack('>4f', reply[1:17])) if isinstance(reply, bytes) and len(reply) >= 17 else reply
        for mid, key in ((XMID_REQ_FILTER_PROFILE, 'filter_profile'), (XMID_REQ_AVAILABLE_FILTER_PROFILES, 'available_filter_profiles')):
            reply = transact(port, reader, mid)
            report[key] = reply.hex() if isinstance(reply, bytes) else reply
    finally:
        transact(port, reader, XMID_GOTO_MEASUREMENT)
    raw, decoded, errors = bytearray(), [], []
    decoder = ImuDecoder(np.eye(3))
    reader = XbusReader()
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        data = port.read(port.in_waiting or 1)
        raw.extend(data)
        for mid, payload in reader.feed(data):
            if mid != XMID_MTDATA2:
                continue
            report.setdefault('stream_data_ids', sorted(f'0x{i:04X}' for i in parse_mtdata2(payload)))
            try:
                decoded.append(decoder.decode(payload, time.monotonic()))
            except ValueError as exc:
                errors.append(str(exc))
    port.close()
    report['packets_decoded'] = len(decoded)
    report['decode_errors'] = sorted(set(errors))
    if len(decoded) > 1:
        report['rate_hz'] = (len(decoded) - 1) / (decoded[-1]['sample_at'] - decoded[0]['sample_at'])
        report['mean_sensor_gravity'] = np.mean([d['gravity'] for d in decoded], axis=0).round(4).tolist()
        report['mean_gyro_rad_s'] = np.mean([d['gyro'] for d in decoded], axis=0).round(5).tolist()
    if args.raw_dump:
        args.raw_dump.write_bytes(bytes(raw))
        report['raw_dump'] = str(args.raw_dump)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
