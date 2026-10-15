#!/usr/bin/env python3
"""Handover get-ups per handover-posture bin and side, per window of iterations; no simulation (#6).

Reads the per-step Handover/* counts the environment logs to tensorboard. The runner logs each
iteration's mean over its policy steps, so mean x steps-per-iteration is that iteration's total.
"""
import argparse
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('--window', type=int, default=10)
p.add_argument('--steps-per-iteration', type=int, help='default: num_steps_per_env from the run.json')
args = p.parse_args()
if args.steps_per_iteration is None:
    args.steps_per_iteration = json.loads((args.run / 'run.json').read_text())['runner']['num_steps_per_env']
accumulator = EventAccumulator(str(args.run), size_guidance={'scalars': 0})
accumulator.Reload()
per_iteration, curriculum, means = {}, {}, {}
for tag in accumulator.Tags()['scalars']:
    if tag.startswith(('Handover/', 'Recovery/')):
        for event in accumulator.Scalars(tag):
            per_iteration.setdefault(event.step, {})[tag] = event.value * args.steps_per_iteration
    elif tag.startswith(('Reward/', 'Train/', 'Policy/', 'Loss/')) and not tag.endswith('/time'):
        # Per-step reward means (per environment) and runner statistics: averaged over each window.
        for event in accumulator.Scalars(tag):
            means.setdefault(event.step, {})[tag] = event.value
    elif tag.startswith('Curriculum/'):
        # Frontier state is logged every step; the iteration mean is its average over that iteration.
        for event in accumulator.Scalars(tag):
            curriculum.setdefault(event.step, {})[tag[len('Curriculum/'):]] = event.value
iterations = sorted(per_iteration)
if not iterations:
    raise SystemExit(f'No Handover/* or Recovery/* scalars in {args.run}')
windows = []
for first in range(0, len(iterations), args.window):
    chunk = iterations[first:first + args.window]
    totals = {}
    for iteration in chunk:
        for tag, value in per_iteration[iteration].items():
            totals[tag] = totals.get(tag, 0.) + value
    row = {'iterations': [chunk[0], chunk[-1]]}
    for bin_name in ('official', 'below_0.2', '0.2_to_0.5', 'at_least_0.5'):
        for side in ('left', 'right'):
            key = f'{bin_name}_{side}'
            counts = {event: round(totals.get(f'Handover/{key}_{event}', 0.))
                      for event in ('started', 'completed', 'stable_get_ups', 'attempt_truncations')}
            counts['get_ups_per_completed'] = counts['stable_get_ups'] / counts['completed'] if counts['completed'] else None
            row[key] = counts
    if curriculum:
        row['curriculum_at_window_end'] = curriculum.get(chunk[-1], {})
    recovery = {tag[len('Recovery/'):]: round(value) for tag, value in totals.items() if tag.startswith('Recovery/')}
    if recovery:
        row['recovery'] = recovery
    values = {}
    for iteration in chunk:
        for tag, value in means.get(iteration, {}).items():
            values.setdefault(tag, []).append(value)
    if values:
        row['window_means'] = {tag: sum(v) / len(v) for tag, v in sorted(values.items())}
    windows.append(row)
result = {'run': str(args.run), 'window_iterations': args.window, 'steps_per_iteration': args.steps_per_iteration,
          'method': 'per-iteration mean of per-step event counts x steps per iteration, summed per window; '
                    'window_means: Reward/* (per-step mean over environments), Train/*, Policy/*, Loss/* averaged over the window',
          'windows': windows}
(args.run / 'handover-bin-rates.json').write_text(json.dumps(result, indent=1) + '\n')
print(json.dumps(result, indent=1))
