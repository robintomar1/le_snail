"""Draw the figures of docs/learning/ (training-curve notes and worksheet) from the exported scalars.

Usage:
    uv run --no-project --with matplotlib --with numpy scripts/plot_training_curves.py \
        docs/learning/training-scalars.json docs/learning/figures

Curves are raw (TensorBoard smoothing 0). A resumed chunk continues its parent's iteration count; each
resume is drawn as a dashed vertical line, because the first iterations after one are distorted.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#e87ba4']
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 10,
    'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.color': INK2,
    'ytick.color': INK2, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'legend.frameon': False, 'lines.linewidth': 1.6, 'font.size': 9,
})

scalars = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


def curve(run, tag):
    points = scalars[run].get(tag, [])
    return [p[0] for p in points], [p[1] for p in points]


def lineage_curve(chunks, tag, combine=None):
    """Concatenated (x, y) segments of one lineage; combine sums several tags."""
    segments = []
    for run in chunks:
        if combine:
            parts = [dict(map(tuple, scalars[run].get(t, []))) for t in combine]
            steps = sorted(set.intersection(*(set(p) for p in parts)))
            segments.append((steps, [sum(p[s] for p in parts) for s in steps]))
        else:
            segments.append(curve(run, tag))
    return segments


def panel(ax, lineages, tag, title=None, log=False, combine=None):
    for (label, chunks), color in zip(lineages, SERIES):
        for i, (x, y) in enumerate(lineage_curve(chunks, tag, combine)):
            ax.plot(x, y, color=color, label=label if i == 0 else None)
    for label, chunks in lineages:
        for run in chunks[1:]:
            ax.axvline(curve(run, 'Loss/value_function')[0][0], color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.set_title(title or tag)
    if log:
        ax.set_yscale('log')


def dashboard(name, lineages, panels, title):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4))
    for ax, spec in zip(axes.flat, panels):
        spec = spec if isinstance(spec, dict) else {'tag': spec}
        panel(ax, lineages, **spec)
    for ax in axes[1]:
        ax.set_xlabel('iteration')
    if len(lineages) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper right', ncol=len(lineages), fontsize=9)
    fig.suptitle(title + '   (dashed line = resumed from a checkpoint)', x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / name, dpi=110)
    plt.close(fig)


OLD = ['Train/mean_reward', 'Train/mean_episode_length', 'Reward/progress', 'Reward/yaw', 'Reward/lateral',
       'Policy/mean_noise_std', 'Loss/value_function', {'tag': 'Loss/learning_rate', 'log': True}]
GETUPS = {'tag': 'stable', 'title': 'Recovery/{left+right}_stable_get_ups',
          'combine': ['Recovery/left_stable_get_ups', 'Recovery/right_stable_get_ups']}
TWO_PORTION = ['Train/mean_reward', 'Train/mean_episode_length', 'Reward/forward', 'Reward/getup_potential',
               GETUPS, 'Recovery/upright_falls', 'Policy/mean_noise_std', 'Loss/value_function']

# Worksheet exercises.
dashboard('ex1-crawl-supported.png', [('crawl-supported-001', ['crawl-supported-001'])],
          [t if t != 'Reward/yaw' else 'Reward/tilt' for t in OLD], 'Exercise 1: crawl-supported-001')
dashboard('ex2-scale-pair.png', [('control', ['scale-control-001-050', 'scale-control-051-100']),
                                 ('treatment', ['scale-treatment-001-050', 'scale-treatment-051-100'])],
          OLD, 'Exercise 2: scale-control vs scale-treatment')
dashboard('ex3-diversity-seeds.png', [('seed 101', ['diversity-seed101-001-050', 'diversity-seed101-051-100']),
                                      ('seed 202', ['diversity-seed202-001-050', 'diversity-seed202-051-100'])],
          OLD, 'Exercise 3: diversity seed 101 vs seed 202')
dashboard('ex4-two-portion-50hz.png', [('two-portion-50hz-001-100', ['two-portion-50hz-001-100'])],
          TWO_PORTION[:6] + ['Policy/mean_noise_std', {'tag': 'Loss/learning_rate', 'log': True}],
          'Exercise 4: two-portion-50hz-001-100')
HYBRID = ['two-portion-hybrid-001-100'] + [f'two-portion-hybrid-002-{n}' for n in (200, 300, 400, 500)]
dashboard('ex5-hybrid-lineage.png', [('hybrid', HYBRID)],
          ['Train/mean_reward', 'Train/mean_episode_length', 'Reward/forward', 'Reward/getup_potential',
           {'tag': 'Recovery/left_stable_get_ups', 'title': 'Recovery/left_stable_get_ups'},
           {'tag': 'Recovery/right_stable_get_ups', 'title': 'Recovery/right_stable_get_ups'},
           'Policy/mean_noise_std', 'Loss/value_function'],
          'Exercise 5: two-portion-hybrid-001 then 002 to iteration 500')
dashboard('ex6-hybrid-003-vs-004.png', [('003 (heading penalty)', ['two-portion-hybrid-003-600']),
                                        ('004 (forward scaled by heading)',
                                         ['two-portion-hybrid-004-550', 'two-portion-hybrid-004-600'])],
          TWO_PORTION, 'Exercise 6: two-portion-hybrid-003 vs 004, both resumed from 002-500')

# Notes on diversity-seed101-051-100.
RUN = 'diversity-seed101-051-100'
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
x, reward = curve(RUN, 'Train/mean_reward')
_, length = curve(RUN, 'Train/mean_episode_length')
for ax, (y, title) in zip(axes, [(reward, 'Train/mean_reward'), (length, 'Train/mean_episode_length (steps)'),
                                 ([r / n for r, n in zip(reward, length)], 'reward per step = reward / length')]):
    ax.plot(x, y, color=SERIES[0])
    ax.set_title(title)
    ax.set_xlabel('iteration')
fig.tight_layout()
fig.savefig(out / 'notes-reward-vs-length.png', dpi=110)
plt.close(fig)

TERMS = ['progress', 'command_rate', 'yaw', 'time', 'tilt', 'lateral']
final = {t: sum(v for _, v in scalars[RUN][f'Reward/{t}'][-5:]) / 5 for t in TERMS}
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.barh(TERMS[::-1], [abs(final[t]) for t in TERMS[::-1]], color=SERIES[0], height=0.55)
for i, t in enumerate(TERMS[::-1]):
    ax.text(abs(final[t]) * 1.15, i, f'{final[t]:+.2g} per step', va='center', fontsize=8, color=INK)
ax.set_xscale('log')
ax.set_xlim(2e-5, 0.6)
ax.set_xlabel('size of the term per step (log scale; sign in the label)')
ax.set_title('Reward terms, mean of the last 5 iterations (protected_contact is 0)')
ax.grid(axis='y', visible=False)
fig.tight_layout()
fig.savefig(out / 'notes-reward-terms.png', dpi=110)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, tag in zip(axes, ['Policy/mean_noise_std', 'Loss/entropy', 'Loss/learning_rate']):
    ax.plot(*curve(RUN, tag), color=SERIES[0])
    ax.set_title(tag)
    ax.set_xlabel('iteration')
axes[2].set_yscale('log')
axes[2].axhline(3e-4, color=INK2, lw=0.8, ls=(0, (3, 3)))
axes[2].text(52, 3.6e-4, 'configured start 3e-4', fontsize=8, color=INK2)
fig.tight_layout()
fig.savefig(out / 'notes-exploration.png', dpi=110)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 3.4))
panel(ax, [('seed 101', ['diversity-seed101-001-050', RUN])], 'Loss/value_function')
ax.annotate('right after the resume:\nall robots restarted together', xy=(50, 0.30), xytext=(58, 0.9),
            fontsize=8, color=INK, arrowprops={'arrowstyle': '->', 'color': INK2})
ax.set_xlabel('iteration')
fig.tight_layout()
fig.savefig(out / 'notes-value-loss.png', dpi=110)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.8))
for (label, run), color in zip([('diversity-seed101-051-100', RUN), ('crawl-straight-001', 'crawl-straight-001'),
                                ('handover-reverse-001-100', 'handover-reverse-001-100'),
                                ('two-portion-hybrid-002-500', 'two-portion-hybrid-002-500'),
                                ('two-portion-hybrid-004-600', 'two-portion-hybrid-004-600')], SERIES):
    x, y = curve(run, 'Policy/mean_noise_std')
    ax.plot(x, y, color=color)
    ax.text(x[-1] + 6, y[-1], label, va='center', fontsize=8, color=INK)
ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.text(560, 1.08, 'std = 1: noise as large as the ±1 action limit', fontsize=8, color=INK2)
ax.set_xlim(0, 790)
ax.set_xlabel('iteration (cumulative over each lineage)')
ax.set_title('Policy/mean_noise_std in five runs')
fig.tight_layout()
fig.savefig(out / 'notes-noise-across-runs.png', dpi=110)
plt.close(fig)
print(f'figures -> {out}')
