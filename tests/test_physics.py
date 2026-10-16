import json
from pathlib import Path
import pytest
from le_snail.physics import load_physics


@pytest.mark.parametrize('key,value', [
    ('drag_force_range_n', [-.1,.3]),
    ('mass_scale_range', [0,1]),
    ('physics_dt', 0),
    ('drag_velocity_scale_m_s', float('nan')),
    ('static_friction_range', [.8,.2]),
])
def test_nonphysical_parameters_rejected_before_simulation(tmp_path, key, value):
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/sim.json').read_text())
    config[key] = value
    path = tmp_path/'sim.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_physics(path)


def test_command_delay_in_ms_is_validated(tmp_path):
    import json
    from le_snail.physics import load_physics
    base = json.loads((Path(__file__).resolve().parents[1] / 'configs/experiments/physics-v2-50hz-001.json').read_text())
    for good in ({'command_delay_ms': 50.}, {'command_delay_ms': 50., 'command_delay_range_ms': [20., 100.]}):
        path = tmp_path / 'good.json'
        path.write_text(json.dumps({**base, **good}))
        load_physics(path)
    for bad in ({'command_delay_ms': -1.}, {'command_delay_ms': 50., 'command_delay_range_ms': [60., 100.]}):
        path = tmp_path / 'bad.json'
        path.write_text(json.dumps({**base, **bad}))
        with pytest.raises(ValueError):
            load_physics(path)
