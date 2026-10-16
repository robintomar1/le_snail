#!/usr/bin/env python3
"""Live Xsens MTi viewer: serves a local web page with orientation, gyro and acceleration plots.

Read-only: it never writes to the device. Open http://localhost:8765 after starting it.
"""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import struct
import threading
import time

from le_snail.hardware import (
    XDI_ACCELERATION, XDI_PACKET_COUNTER, XDI_QUATERNION, XDI_RATE_OF_TURN, XDI_SAMPLE_TIME_FINE,
    XMID_ERROR, XMID_MTDATA2, XRV_DATAOVERFLOW, XbusReader, enable_low_latency, parse_mtdata2,
)

XDI_STATUS_WORD = 0xE020


class ImuFeed:
    def __init__(self, port, baudrate, history=1000):
        import serial
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=0.05)
        enable_low_latency(self.serial)
        self.serial.reset_input_buffer()
        self.reader = XbusReader(max_consecutive_errors=10_000)
        self.samples = deque(maxlen=history)
        self.index = 0
        self.overflows = 0
        self.errors = []
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            try:
                data = self.serial.read(self.serial.in_waiting or 1)
            except Exception as exc:
                self.errors.append(f'serial: {exc}')
                time.sleep(1)
                continue
            received = time.monotonic()
            for mid, payload in self.reader.feed(data):
                if mid == XMID_ERROR:
                    if payload[:1] == bytes((XRV_DATAOVERFLOW,)):
                        self.overflows += 1
                    else:
                        self.errors.append(f'device error 0x{payload.hex()}')
                    continue
                if mid != XMID_MTDATA2:
                    continue
                try:
                    sample = self._decode(parse_mtdata2(payload), received)
                except (KeyError, ValueError, struct.error):
                    continue
                with self.lock:
                    self.index += 1
                    sample['i'] = self.index
                    self.samples.append(sample)

    @staticmethod
    def _decode(items, received):
        def get(xdi, fmt):
            for key, value in items.items():
                if key & 0xFFF0 == xdi:
                    return struct.unpack(fmt, value)
            return None
        w, x, y, z = get(XDI_QUATERNION, '>4f') or (math.nan,) * 4
        roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        gyro = get(XDI_RATE_OF_TURN, '>3f') or (math.nan,) * 3
        acc = get(XDI_ACCELERATION, '>3f') or (math.nan,) * 3
        status = (get(XDI_STATUS_WORD, '>I') or (0,))[0]
        return {
            't': received, 'counter': (get(XDI_PACKET_COUNTER, '>H') or (None,))[0],
            'stf': (get(XDI_SAMPLE_TIME_FINE, '>I') or (None,))[0],
            'q': [w, x, y, z], 'rpy': [roll, pitch, yaw], 'gyro': list(gyro), 'acc': list(acc),
            'selftest': bool(status & 1), 'filter_ok': bool(status & 2),
        }

    def since(self, index):
        with self.lock:
            return [s for s in self.samples if s['i'] > index]


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>IMU Live View</title>
<style>
:root{--bg:#f4f5f2;--panel:#fff;--ink:#1d2126;--muted:#5b636d;--rule:#d9dbd6;--x:#c2410c;--y:#15803d;--z:#1d4ed8;--accent:#b88a00;--ok:#15803d;--bad:#b91c1c;
--mono:ui-monospace,"SFMono-Regular",Menlo,monospace;--sans:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root{--bg:#15181c;--panel:#1d2126;--ink:#e8e9e4;--muted:#a2a9b1;--rule:#343a42;--x:#fb923c;--y:#4ade80;--z:#60a5fa;--accent:#f0c23a;--ok:#4ade80;--bad:#f87171}}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 var(--sans);padding:16px max(16px,env(safe-area-inset-left))}
.wrap{max-width:1180px;margin:0 auto;display:grid;gap:16px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 20px;border-bottom:3px solid var(--accent);padding-bottom:10px}
h1{font-size:1.4rem;margin:0}.sub{color:var(--muted);font-family:var(--mono);font-size:.8rem}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px 12px}
.tile .k{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.tile .v{font-family:var(--mono);font-size:1.25rem;font-variant-numeric:tabular-nums}
.ok{color:var(--ok)}.bad{color:var(--bad)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:16px}
@media (max-width:860px){.grid{grid-template-columns:1fr}}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px 12px;min-width:0}
.panel h2{font-size:.85rem;margin:0 0 4px;font-family:var(--mono);color:var(--muted);font-weight:500;display:flex;justify-content:space-between;gap:8px}
canvas{width:100%;display:block}.legend span{margin-left:10px}.lx{color:var(--x)}.ly{color:var(--y)}.lz{color:var(--z)}
.charts{display:grid;gap:12px}
</style></head><body><div class="wrap">
<header><h1>IMU live view</h1><span class="sub" id="dev">Xsens MTi · waiting for data…</span></header>
<div class="tiles">
 <div class="tile"><div class="k">Sample rate</div><div class="v" id="rate">–</div></div>
 <div class="tile"><div class="k">Arrival gap (max, 2 s)</div><div class="v" id="gap">–</div></div>
 <div class="tile"><div class="k">Filter</div><div class="v" id="filt">–</div></div>
 <div class="tile"><div class="k">Tilt from vertical</div><div class="v" id="tilt">–</div></div>
 <div class="tile"><div class="k">|accel|</div><div class="v" id="amag">–</div></div>
 <div class="tile"><div class="k">Packets / overflows</div><div class="v" id="pk">–</div></div>
</div>
<div class="grid">
 <div class="charts">
  <div class="panel"><h2>Orientation (deg) <span class="legend"><span class="lx">roll</span><span class="ly">pitch</span><span class="lz">yaw</span></span></h2><canvas id="c0" height="170"></canvas></div>
  <div class="panel"><h2>Rate of turn (rad/s) <span class="legend"><span class="lx">x</span><span class="ly">y</span><span class="lz">z</span></span></h2><canvas id="c1" height="170"></canvas></div>
  <div class="panel"><h2>Acceleration (m/s²) <span class="legend"><span class="lx">x</span><span class="ly">y</span><span class="lz">z</span></span></h2><canvas id="c2" height="170"></canvas></div>
 </div>
 <div class="panel"><h2>Board attitude (sensor frame) <span id="rpy"></span></h2><canvas id="cube" height="320"></canvas>
  <p class="sub">Arrow = sensor x (red), y (green), z (blue). Heading is free-running (no magnetometer).</p></div>
</div></div>
<script>
const WINDOW_S = 10, css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let buf = [], overflow = 0, lastI = 0;
const ev = new EventSource('/events');
ev.onmessage = e => { const m = JSON.parse(e.data); overflow = m.overflows; for (const s of m.samples) { buf.push(s); lastI = s.i; }
  const tmin = buf.length ? buf[buf.length-1].t - WINDOW_S : 0; while (buf.length && buf[0].t < tmin) buf.shift(); if (m.device) document.getElementById('dev').textContent = m.device; };
function fit(c){ const r = c.getBoundingClientRect(), d = devicePixelRatio || 1; if (c.width !== Math.round(r.width*d)) { c.width = Math.round(r.width*d); c.height = Math.round(c.getAttribute('height')*d); } return d; }
function chart(id, key){ const c = document.getElementById(id), d = fit(c), g = c.getContext('2d'), W = c.width, H = c.height;
  g.clearRect(0,0,W,H); if (buf.length < 2) return; const t1 = buf[buf.length-1].t, t0 = t1 - WINDOW_S;
  let lo = Infinity, hi = -Infinity; for (const s of buf) for (const v of s[key]) { if (isFinite(v)) { lo = Math.min(lo,v); hi = Math.max(hi,v); } }
  if (hi - lo < 1e-3) { hi += 0.5; lo -= 0.5; } const pad = (hi-lo)*0.1; lo -= pad; hi += pad;
  const X = t => (t-t0)/WINDOW_S*(W-50*d)+44*d, Y = v => H-12*d-(v-lo)/(hi-lo)*(H-24*d);
  g.strokeStyle = css('--rule'); g.fillStyle = css('--muted'); g.font = `${11*d}px ui-monospace,monospace`; g.lineWidth = d;
  for (let k=0;k<=4;k++){ const v = lo+(hi-lo)*k/4, y = Y(v); g.beginPath(); g.moveTo(44*d,y); g.lineTo(W,y); g.stroke(); g.fillText(v.toFixed(Math.abs(hi-lo)<2?3:1), 2*d, y+4*d); }
  [['--x',0],['--y',1],['--z',2]].forEach(([col,j]) => { g.strokeStyle = css(col); g.lineWidth = 1.6*d; g.beginPath(); let started=false;
    for (const s of buf){ const v = s[key][j]; if (!isFinite(v)) continue; const x = X(s.t), y = Y(v); started ? g.lineTo(x,y) : g.moveTo(x,y); started = true; } g.stroke(); }); }
function quatMat([w,x,y,z]){ return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]; }
function cube(){ const c = document.getElementById('cube'), d = fit(c), g = c.getContext('2d'), W = c.width, H = c.height; g.clearRect(0,0,W,H); if (!buf.length) return;
  const s = buf[buf.length-1], R = quatMat(s.q), yawR = s.rpy[2]*Math.PI/180, cy = Math.cos(-yawR), sy = Math.sin(-yawR);
  const rot = p => { const v = [0,1,2].map(i => R[i][0]*p[0]+R[i][1]*p[1]+R[i][2]*p[2]); return [cy*v[0]-sy*v[1], sy*v[0]+cy*v[1], v[2]]; };
  const az = 0.75, el = 0.45, ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el), sc = Math.min(W,H)*0.28;
  const P = v => { const xr = ca*v[0]-sa*v[1], yr = sa*v[0]+ca*v[1]; return [W/2+sc*xr, H/2-sc*(ce*v[2]-se*yr)]; };
  g.strokeStyle = css('--rule'); g.lineWidth = d; for (let k=-2;k<=2;k++){ let a=P([k*0.5,-1,-1.1]),b=P([k*0.5,1,-1.1]); g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke(); a=P([-1,k*0.5,-1.1]); b=P([1,k*0.5,-1.1]); g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke(); }
  const h=[0.9,0.6,0.15], V=[]; for (const sx of [-1,1]) for (const syy of [-1,1]) for (const sz of [-1,1]) V.push(P(rot([sx*h[0],syy*h[1],sz*h[2]])));
  g.strokeStyle = css('--ink'); g.lineWidth = 1.5*d; [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]].forEach(([a,b]) => { g.beginPath(); g.moveTo(...V[a]); g.lineTo(...V[b]); g.stroke(); });
  [['--x',[1.4,0,0]],['--y',[0,1.1,0]],['--z',[0,0,0.9]]].forEach(([col,p]) => { const o = P(rot([0,0,0])), e = P(rot(p)); g.strokeStyle = css(col); g.lineWidth = 3*d; g.beginPath(); g.moveTo(...o); g.lineTo(...e); g.stroke(); });
  document.getElementById('rpy').textContent = `r ${s.rpy[0].toFixed(1)}° p ${s.rpy[1].toFixed(1)}°`; }
function tiles(){ if (buf.length < 2) return; const s = buf[buf.length-1], recent = buf.filter(b => b.t > s.t-2);
  const rate = recent.length > 1 ? (recent.length-1)/(recent[recent.length-1].stf - recent[0].stf)*1e4 : 0;
  let gap = 0; for (let i=1;i<recent.length;i++) gap = Math.max(gap, recent[i].t-recent[i-1].t);
  const [w,x,y,z] = s.q, upz = 1-2*(x*x+y*y), am = Math.hypot(...s.acc);
  document.getElementById('rate').textContent = rate.toFixed(1)+' Hz';
  document.getElementById('gap').textContent = (gap*1000).toFixed(0)+' ms';
  const f = document.getElementById('filt'); f.textContent = s.filter_ok ? 'valid' : 'NOT valid'; f.className = 'v '+(s.filter_ok?'ok':'bad');
  document.getElementById('tilt').textContent = (Math.acos(Math.max(-1,Math.min(1,upz)))*180/Math.PI).toFixed(1)+'°';
  document.getElementById('amag').textContent = am.toFixed(2);
  document.getElementById('pk').textContent = `${s.counter} / ${overflow}`; }
(function loop(){ chart('c0','rpy'); chart('c1','gyro'); chart('c2','acc'); cube(); tiles(); requestAnimationFrame(loop); })();
</script></body></html>"""


def make_handler(feed, device):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == '/':
                body = PAGE.encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == '/events':
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                last = 0
                try:
                    while True:
                        new = feed.since(last)
                        if new:
                            last = new[-1]['i']
                        msg = {'samples': new, 'overflows': feed.overflows, 'device': device}
                        self.wfile.write(f'data: {json.dumps(msg)}\n\n'.encode())
                        self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_error(404)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', required=True, help='serial port, e.g. /dev/serial/by-id/usb-Xsens_...')
    parser.add_argument('--baudrate', type=int, default=115200)
    parser.add_argument('--http-port', type=int, default=8765)
    parser.add_argument('--host', default='127.0.0.1', help='bind address (default: this machine only)')
    args = parser.parse_args()
    feed = ImuFeed(args.port, args.baudrate)
    device = f'Xsens MTi · {args.port.rsplit("/", 1)[-1]} · {args.baudrate} baud'
    server = ThreadingHTTPServer((args.host, args.http_port), make_handler(feed, device))
    print(f'IMU viewer at http://localhost:{args.http_port}  (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
