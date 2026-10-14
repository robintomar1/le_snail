"""Exercise saved-data timing and contradiction gates without Isaac/Kit."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest


@pytest.mark.parametrize('case', ['distinct', 'conflict', 'aliased_desired'])
def test_saved_actor24_gate_accepts_consistency_and_rejects_conflict(tmp_path, case):
    conflict = case == 'conflict'
    contract=json.loads(Path('configs/policy.json').read_text())
    home=np.array(contract['home_rad'],dtype=np.float32)
    scale=np.array(contract['action_scale_rad'],dtype=np.float32)
    source=tmp_path/'input';source.mkdir()
    groups=[]
    for number,(side,variant,n) in enumerate([('left',9,1),('left',34,1),('right',48,1),('right',62,1),('upright',-1,4)]):
        obs=np.zeros((1,n,24),dtype=np.float32)
        obs[0,:,15]=number+np.arange(n)*.2
        if case != 'distinct' and number<2: obs[0,:,15]=0
        pre=np.broadcast_to(home,(1,n,6)).copy()
        act=np.ones((1,n,6),dtype=np.float32)*(-1 if conflict and number==1 else 1)
        if case == 'aliased_desired' and number == 1: act *= 2
        desired=np.clip(home+np.clip(act,-1,1)*scale,contract['lower_rad'],contract['upper_rad'])
        post=pre+np.clip(desired-pre,-.05,.05)
        stem=f'g{number}'
        np.savez_compressed(source/f'{stem}.npz',observations=obs,desired_action=act,pre_target=pre,post_applied=post,joint_q=pre,filtered_velocity=np.zeros_like(pre),pre_applied=pre,active=np.ones((1,n),dtype=bool),phase=np.zeros((1,n),dtype=int))
        groups.append({'stem':stem,'side':side,'repeat':0,'variants':[variant]*n,'successes':n,'trials':[{'env':i,'success':True,'acceptance_end_s':.05}for i in range(n)]})
    (source/'report.json').write_text(json.dumps({'collection_complete':True,'positive_control_passed':True,'policy_dt':.05,'groups':groups,'ambiguity_preregistered':{'test_fixture':True}}))
    output=tmp_path/'output'
    process=subprocess.run([sys.executable,'scripts/validate_recovery_demonstrations.py','--input',str(source),'--output',str(output)],capture_output=True,text=True)
    assert process.returncode==(2 if conflict else 0),process.stdout+process.stderr
    result=json.loads((output/'validation.json').read_text())
    assert result['dataset_valid'] is not conflict
    assert bool(result['conflicting_pairs']) is conflict
    assert (output/'dataset.npz').exists() is not conflict
