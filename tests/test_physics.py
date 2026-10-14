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
