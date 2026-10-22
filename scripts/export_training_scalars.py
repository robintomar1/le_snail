"""Export the scalar curves of RSL-RL TensorBoard runs to one compact JSON file.

Usage (TensorBoard is not a project dependency):
    uv run --no-project --with tensorboard==2.18.0 --with "setuptools<70" \
        scripts/export_training_scalars.py <runs_dir> docs/learning/training-scalars.json

<runs_dir> holds one folder per run with its events.out.tfevents.* file, as the GPU host's runs/ does.
Output: {run: {tag: [[iteration, value], ...]}}, values rounded to 4 significant figures.
"""
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
curves = {}
for run in sorted(p for p in runs_dir.iterdir() if any(p.glob('events.out.tfevents.*'))):
    events = EventAccumulator(str(run), size_guidance={'scalars': 0})
    events.Reload()
    curves[run.name] = {tag: [[e.step, float(f'{e.value:.4g}')] for e in events.Scalars(tag)]
                        for tag in events.Tags()['scalars']}
output.write_text(json.dumps(curves, separators=(',', ':')))
print(f'{len(curves)} runs -> {output} ({output.stat().st_size // 1024} KiB)')
