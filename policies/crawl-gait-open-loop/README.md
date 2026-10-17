# Open-loop crawl gait (hybrid-001)

The crawl that first moved the real arm forward (2026-10-17 03:11 UTC, arm 1; frozen as `runs/best_forward_move_1/`).

- `hybrid-001-targets-50hz/` — the hybrid-001 controller's joint targets exported from sim by `scripts/export_hybrid_targets.py` (crawl, left and right episodes; 50 Hz, post-slew, URDF radians; `crawl.npz` sha256 44f5a6f8…).
- `crawl-10s.npz`, `crawl-30s.npz` — the first 10 s / 30 s of the crawl episode converted to servo-tick offsets for arm 1 by `scripts/targets_to_motion.py` with `configs/joint-maps/so101-esp32-arm-2026-10-17.json` (start ticks 2176 1747 2604 1714 2076). Played with `scripts/actuator_log.py --motion … --start-ticks … --ramp-s 5`.
