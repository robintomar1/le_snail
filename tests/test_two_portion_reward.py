"""Two-portion reward (#6): forward = new ground while upright; standing up = progress potential + standing."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from le_snail.physics import load_physics

torch = pytest.importorskip('torch')
from le_snail.rewards import ProgressBank, ramp, two_portion_terms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = {'standing_per_s': .25, 'standing_ramp_cos': [-.1, .8], 'forward_per_m': 30., 'forward_gate_cos': [.5, .8],
          'getup_progress_weight': 2., 'effort_weight': .02}


def line_bank(tmp_path, points=101):
    """Bank states along one standardized axis with phase rising 0 -> 1."""
    states = np.zeros((points, 9), dtype=np.float32)
    states[:, 0] = np.linspace(0., 3., points)
    path = tmp_path / 'bank.npz'
    np.savez(path, bank_states=states, phases=np.linspace(0., 1., points, dtype=np.float32), mean=np.zeros(9, np.float32),
             std=np.ones(9, np.float32), sigma=np.float32(.3), k=np.int64(16), shrink=np.float32(1.))
    return ProgressBank(path, 'cpu')


def step(bank, upright, x, x_record, previous, terminated=False, rate=0., features=None):
    n = len(upright)
    features = torch.full((n, 9), 50.) if features is None else features   # far from the bank: progress ~0
    return two_portion_terms(CONFIG, bank, upright=torch.tensor(upright), features=features, x=torch.tensor(x),
                             x_record=torch.tensor(x_record), previous_phi=torch.tensor(previous),
                             terminated=torch.tensor([terminated] * n), command_rate=torch.full((n, 6), rate), dt=.02, gamma=.998)


def test_progress_rises_along_the_bank_and_vanishes_far_away(tmp_path):
    bank = line_bank(tmp_path)
    along = torch.zeros((31, 9))
    along[:, 0] = torch.linspace(.3, 2.7, 31)
    progress = bank.progress(along)
    assert bool((progress[1:] > progress[:-1]).all())
    assert float(progress[15]) == pytest.approx(.5, abs=.06)
    assert float(bank.progress(torch.full((1, 9), 5.))[0]) < 1e-6


def test_forward_pays_only_new_ground_while_upright():
    x_record, credits = 0., []
    for x, up in ((.10, .95), (.05, .95), (.12, .95), (.20, -.1), (.25, .95)):
        terms, _, record = step(_far_bank(), [up], [x], [x_record], [float('nan')])
        credits.append(float(terms['forward'][0]))
        x_record = float(record[0])
    # Ground gained while fallen (0.12 -> 0.20) is never paid, even after standing up again.
    assert credits == pytest.approx([3., 0., .6, 0., 1.5], abs=1e-5)


def _far_bank():
    class Far:
        def progress(self, features):
            return torch.zeros(len(features))
    return Far()


def test_effort_counts_only_while_upright_and_standing_is_convex():
    terms, phi, _ = step(_far_bank(), [-.1, .35, .95], [0.] * 3, [0.] * 3, [float('nan')] * 3, rate=1.)
    assert terms['effort'].tolist()[:2] == [0., 0.] and float(terms['effort'][2]) == pytest.approx(-.02 * .02 * 6)
    assert terms['standing'].tolist() == pytest.approx([0., .25 * .02 * .25, .25 * .02])
    assert phi.tolist() == pytest.approx([0., .25, 1.])


def test_shaping_is_zero_on_first_step_uses_zero_terminal_potential_and_nets_zero_over_a_loop():
    terms, _, _ = step(_far_bank(), [.95], [0.], [0.], [float('nan')])
    assert float(terms['getup_progress'][0]) == 0.
    terms, _, _ = step(_far_bank(), [.95], [0.], [0.], [1.], terminated=True)
    assert float(terms['getup_progress'][0]) == pytest.approx(-2.)
    gamma, previous, total = .998, 0., 0.
    for t, up in enumerate((.35, .8, .35, -.1)):
        terms, phi, _ = step(_far_bank(), [up], [0.], [0.], [previous])
        total += gamma ** t * float(terms['getup_progress'][0])
        previous = float(phi[0])
    assert total == pytest.approx(0., abs=1e-6)


def test_two_portion_config_validation(tmp_path):
    config = json.loads((ROOT / 'configs/experiments/two-portion-001.json').read_text())
    assert load_physics(ROOT / 'configs/experiments/two-portion-001.json')['reward']['type'] == 'two_portion'
    for key, value in (('type', 'old'), ('standing_per_s', -1.), ('forward_gate_cos', [.8, .5]), ('getup_progress_bank_sha256', 'x')):
        broken = json.loads(json.dumps(config))
        broken['reward'][key] = value
        path = tmp_path / 'broken.json'
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            load_physics(path)


def test_ramp_clamps():
    assert ramp(torch.tensor([-1., .35, 2.]), -.1, .8).tolist() == pytest.approx([0., .5, 1.])


def test_forward_requires_crawl_mode_when_configured():
    config = dict(CONFIG, forward_requires_crawl_mode=True)
    for crawling, expected in ((False, 0.), (True, 3.)):
        terms, _, record = two_portion_terms(config, _far_bank(), upright=torch.tensor([.95]), features=torch.full((1, 9), 50.),
                                             x=torch.tensor([.1]), x_record=torch.tensor([0.]), previous_phi=torch.tensor([float('nan')]),
                                             terminated=torch.tensor([False]), command_rate=torch.zeros((1, 6)), dt=.02, gamma=.998,
                                             crawling=torch.tensor([crawling]))
        assert float(terms['forward'][0]) == pytest.approx(expected)
        assert float(record[0]) == pytest.approx(.1)   # the record advances even when nothing is paid


# Refinements (#15): roll-based uprightness with a pitch bound, lane, heading, non-support contact.
REFINED = {**CONFIG, 'uprightness': 'roll_pitch', 'pitch_bound_deg': [45., 60.], 'lane_per_s': .5, 'lane_band_m': [.05, .15],
           'heading_per_s': .5, 'heading_band_deg': [30., 90.], 'contact_per_s': 4.}


def _gravity(roll_deg, pitch_deg):
    """Unit gravity in the base frame for a base pitched then rolled (roll about x, pitch about y)."""
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    return torch.tensor([[-math.sin(p), math.cos(p) * math.sin(r), -math.cos(p) * math.cos(r)]])


def _refined(gravity, lateral=0., yaw=0., contact=False, config=REFINED):
    terms, _, _ = two_portion_terms(config, _far_bank(), upright=-gravity[:, 2], features=torch.zeros((1, 9)), x=torch.tensor([.1]),
                                    x_record=torch.tensor([0.]), previous_phi=torch.tensor([float('nan')]), terminated=torch.tensor([False]),
                                    command_rate=torch.zeros((1, 6)), dt=.02, gamma=.998, crawling=None, gravity=gravity,
                                    lateral=torch.tensor([lateral]), yaw=torch.tensor([yaw]), other_contact=torch.tensor([contact]))
    return {k: float(v[0]) for k, v in terms.items()}


def test_pitch_rocking_within_the_bound_keeps_full_standing_and_forward_credit():
    for pitch in (0., 30., 44.):
        terms = _refined(_gravity(0., pitch))
        assert terms['standing'] == pytest.approx(.25 * .02) and terms['forward'] == pytest.approx(3.)
    assert 0. < _refined(_gravity(0., 52.))['forward'] < 3.
    assert _refined(_gravity(0., 61.))['forward'] == 0.
    # The tilt cosine at 44 deg pitch (0.72) would have cut the forward gate to 0.73.
    assert _refined(_gravity(0., 44.), config=CONFIG)['forward'] == pytest.approx(3. * (math.cos(math.radians(44)) - .5) / .3, abs=1e-5)


def test_roll_still_costs_uprightness():
    assert _refined(_gravity(40., 0.))['forward'] == pytest.approx(3. * (math.cos(math.radians(40)) - .5) / .3, abs=1e-5)
    # Lying on a side earns exactly what reward 002 paid there (the pitch bound only changes front-to-back tilt).
    assert _refined(_gravity(90., 0.))['standing'] == pytest.approx(_refined(_gravity(90., 0.), config=CONFIG)['standing'])


def test_lane_heading_and_contact_terms():
    upright = _gravity(0., 20.)
    assert _refined(upright, lateral=.04)['lane'] == 0.
    assert _refined(upright, lateral=-.10)['lane'] == pytest.approx(-.5 * .02 * .5)
    assert _refined(upright, lateral=.2)['lane'] == pytest.approx(-.5 * .02)
    assert _refined(upright, yaw=math.radians(25))['heading'] == 0.
    assert _refined(upright, yaw=-math.radians(60))['heading'] == pytest.approx(-.5 * .02 * .5)
    assert _refined(upright, contact=True)['contact'] == pytest.approx(-4. * .02)
    lying = _gravity(90., 0.)
    assert _refined(lying, yaw=2., contact=True)['heading'] == 0. and _refined(lying, contact=True)['contact'] == 0.


def test_heading_can_be_limited_to_crawl_mode():
    config = {**REFINED, 'heading_per_s': 1., 'heading_band_deg': [15., 60.], 'heading_requires_crawl_mode': True}
    upright = _gravity(0., 20.)

    def heading(crawling):
        terms, _, _ = two_portion_terms(config, _far_bank(), upright=-upright[:, 2], features=torch.zeros((1, 9)), x=torch.tensor([.1]),
                                        x_record=torch.tensor([0.]), previous_phi=torch.tensor([float('nan')]), terminated=torch.tensor([False]),
                                        command_rate=torch.zeros((1, 6)), dt=.02, gamma=.998, crawling=torch.tensor([crawling]), gravity=upright,
                                        lateral=torch.tensor([0.]), yaw=torch.tensor([-math.radians(75)]), other_contact=torch.tensor([False]))
        return float(terms['heading'][0])
    assert heading(False) == 0.
    assert heading(True) == pytest.approx(-1. * .02)


def test_configs_without_refinements_are_unchanged():
    terms = _refined(_gravity(0., 20.), config=CONFIG)
    assert set(terms) == {'getup_progress', 'standing', 'forward', 'effort'}
