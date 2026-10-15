#!/usr/bin/env python3
"""Summarize the handover Phase 1 matched pair from saved artifacts; no simulation (#6).

Reads each arm's training run and its 15-trial evaluation, then writes per-trial tables, five-trial
contact sheets and comparison.json under --output. Numerical results are not a gait review: the
videos still need to be watched for rolling, invalid contact and teacher artifacts.
"""
import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--control', type=Path, default=Path('runs/handover-control-001-050'))
p.add_argument('--treatment', type=Path, default=Path('runs/handover-treatment-001-050'))
p.add_argument('--control-eval', type=Path, default=Path('runs/evaluation-handover-control-050-development-5100'))
p.add_argument('--treatment-eval', type=Path, default=Path('runs/evaluation-handover-treatment-050-development-5100'))
p.add_argument('--no-sheets', action='store_true')
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def training(run, log):
    text = log.read_text(errors='replace') if log.exists() else ''
    iterations = [int(m) for m in re.findall(r'Learning iteration (\d+)/\d+', text)]
    learner = [float(m) for m in re.findall(r'Mean learner_fraction loss:\s+([0-9.]+)', text)]
    speed = load(run / 'speed-diagnostics.json') or {}
    actions = load(run / 'action_diagnostics.json') or {}
    return {'run': str(run), 'exit_code': (log.with_suffix('.exit').read_text().strip() if log.with_suffix('.exit').exists() else None),
            'iterations_logged': len(iterations), 'first_iteration': min(iterations, default=None), 'last_iteration': max(iterations, default=None),
            'export_parity': load(run / 'export-parity.json'),
            'recovery_diagnostics_official_side_starts': load(run / 'recovery-diagnostics.json'),
            'handover_diagnostics': load(run / 'handover-diagnostics.json'),
            'learner_fraction': {'n': len(learner), 'min': min(learner, default=None), 'max': max(learner, default=None)},
            'speed': {k: speed.get(k) for k in ('over_limit_environment_time_fraction', 'peak_abs_velocity_rad_s',
                                                 'longest_driven_proxy_over_limit_s', 'flag_over_one_percent_time',
                                                 'flag_sustained_driven_proxy')},
            'action_fractions_all_joints': (actions.get('all') or {}).get('fractions_all_joints'),
            'exported_policy': str(run / 'exported' / 'policy.pt')}


def evaluation(run, sheets):
    report = load(run / 'report.json')
    if report is None:
        return None
    rows, indices = [], {}
    for index, trial in enumerate(report['trials']):
        kind = trial['kind']
        number = indices.get(kind, 0)
        indices[kind] = number + 1
        stem = f'{kind}-{number}'
        trace = [json.loads(line) for line in (run / f'{stem}.jsonl').read_text().splitlines()]
        last = trace[-1] if trace else {}
        rows.append({'trial': stem, 'seed': report['seed'] + index, 'success': trial['success'], 'reason': trial['reason'],
                     'time_s': last.get('time_s'), 'forward_m': last.get('forward_m'), 'lateral_m': last.get('lateral_m'),
                     'max_abs_lateral_m': max((abs(r['lateral_m']) for r in trace), default=0.),
                     'yaw_from_start_deg': math.degrees(last.get('yaw_from_start_rad', 0.)),
                     'max_upright_cos': max((r['upright_cos'] for r in trace), default=None),
                     'final_upright_cos': last.get('upright_cos'),
                     'stable_upright_time_s': trial['recovery_time_s'], 'video': str(run / f'{stem}.mp4')})
    sheet_paths = {}
    if sheets:
        for kind in ('crawl', 'left', 'right'):
            videos = [run / f'{kind}-{i}.mp4' for i in range(5)]
            if not all(v.exists() for v in videos):
                continue
            out = args.output / f'{run.name}-{kind}-contact.png'
            command, filters = ['ffmpeg', '-v', 'error', '-y'], []
            for i, video in enumerate(videos):
                duration = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                                                          '-of', 'default=noprint_wrappers=1:nokey=1', str(video)], text=True))
                command += ['-i', str(video)]
                filters.append(f'[{i}:v]fps={4 / duration},scale=320:-1,tile=4x1[row{i}]')
            filters.append(''.join(f'[row{i}]' for i in range(5)) + 'vstack=inputs=5[out]')
            subprocess.run(command + ['-filter_complex', ';'.join(filters), '-map', '[out]', '-frames:v', '1', str(out)], check=True)
            sheet_paths[kind] = str(out)
    with (args.output / f'{run.name}-trials.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    speed = report.get('speed_diagnostics') or {}
    return {'run': str(run), 'seed': report['seed'], 'groups': report['groups'], 'passed': report['passed'],
            'side_stable_upright': {k: sum(r['stable_upright_time_s'] is not None for r in rows if r['trial'].startswith(k)) for k in ('left', 'right')},
            'speed': {k: speed.get(k) for k in ('over_limit_environment_time_fraction', 'peak_abs_velocity_rad_s',
                                                 'longest_driven_proxy_over_limit_s', 'flag_over_one_percent_time',
                                                 'flag_sustained_driven_proxy')},
            'trials': rows, 'contact_sheets': sheet_paths}


arms = {}
for name, run, ev in (('control', args.control, args.control_eval), ('treatment', args.treatment, args.treatment_eval)):
    arms[name] = {'training': training(run, Path('.setup') / f'{run.name}.log'), 'evaluation': evaluation(ev, not args.no_sheets)}


def training_get_ups(arm):
    total = 0
    official = arm['training']['recovery_diagnostics_official_side_starts'] or {}
    total += sum(side.get('reached_stable_upright', 0) for side in official.get('sides', {}).values())
    bins = (arm['training']['handover_diagnostics'] or {}).get('bins', {})
    handover = {b: sum(side['reached_stable_upright'] for side in sides.values()) for b, sides in bins.items() if b != 'official_start'}
    return total, handover


summary = {}
for name, arm in arms.items():
    official, handover = training_get_ups(arm)
    ev = arm['evaluation']
    summary[name] = {'training_official_side_get_ups': official, 'training_handover_get_ups_by_bin': handover,
                     'evaluation_crawl_successes': ev['groups']['crawl']['successes'] if ev else None,
                     'evaluation_left_successes': ev['groups']['left']['successes'] if ev else None,
                     'evaluation_right_successes': ev['groups']['right']['successes'] if ev else None,
                     'evaluation_side_stable_upright': ev['side_stable_upright'] if ev else None,
                     'export_parity_passed': (arm['training']['export_parity'] or {}).get('passed'),
                     'evaluation_over_cap_flag': ev['speed'].get('flag_over_one_percent_time') if ev else None}
result = {'schema_version': 1, 'note': 'Development evidence on seeds 5100-5114, not qualification; visual gait review still required.',
          'summary': summary, 'arms': arms}
(args.output / 'comparison.json').write_text(json.dumps(result, indent=1) + '\n')
print(json.dumps(summary, indent=1))
