#!/usr/bin/env python3
"""Audit imported mass/limits and record new nominal rest poses; never train."""
import argparse
import json
from pathlib import Path
import sys
import traceback
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--physics', default='configs/experiments/physics-v2-bootstrap-001.json')
p.add_argument('--asset', default='assets/so101/snail.urdf')
p.add_argument('--contract', default='configs/policy.json')
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.output.exists():
    p.error('--output must be new')
args.output.mkdir(parents=True)
app = AppLauncher(args).app
env = None
try:
    import numpy as np
    import torch
    import xml.etree.ElementTree as ET
    from le_snail.bundle import file_hash
    from le_snail.physics import combine_box_payload
    from le_snail.sim.env import SnailEnv, make_config
    env = SnailEnv(make_config(args.asset,args.contract,args.physics,num_envs=64,stage='recover',evaluation=True,seed=4242,device=args.device or 'cuda:0'))
    view = env.robot.root_physx_view
    report = {'parameters_measured':False, 'checks':[], 'settled_starts':{},
              'provenance':{f:file_hash(f) for f in [args.asset,args.contract,args.physics,__file__,'src/le_snail/sim/env.py','src/le_snail/physics.py']}}
    def check(name, passed, **details):
        report['checks'].append(dict(name=name,passed=bool(passed),**details))
        if not passed:
            (args.output/'report.json').write_text(json.dumps(report,indent=2))
            raise AssertionError(name)
    def array(x):return x.detach().cpu().numpy()
    report['body_names']=env.robot.body_names
    report['imported_masses_kg']=array(env.imported_masses[0]).tolist()
    report['imported_inertias_kg_m2']=array(env.imported_inertias[0]).tolist()
    report['imported_coms_xyzw']=array(env.imported_coms[0]).tolist()
    report['combined_nominal_masses_kg']=array(env.nominal_masses[0]).tolist()
    report['combined_nominal_inertias_kg_m2']=array(env.nominal_inertias[0]).tolist()
    report['combined_nominal_coms_xyzw']=array(env.nominal_coms[0]).tolist()
    upstream=ET.parse('assets/so101/upstream/so101_new_calib.urdf').getroot()
    for i,name in enumerate(env.robot.body_names):
        link=upstream.find("link[@name='"+name+"']/inertial")
        mass=float(link.find('mass').attrib['value'])
        center=np.array([float(v) for v in link.find('origin').attrib['xyz'].split()])
        assert all(float(v)==0 for v in link.find('origin').attrib['rpy'].split())
        a={k:float(v) for k,v in link.find('inertia').attrib.items()}
        inertia=np.array([[a['ixx'],a['ixy'],a['ixz']],[a['ixy'],a['iyy'],a['iyz']],[a['ixz'],a['iyz'],a['izz']]])
        check('upstream_import_'+name, np.isclose(float(env.imported_masses[0,i]),mass,atol=2e-8) and np.allclose(array(env.imported_coms[0,i,:3]),center,atol=2e-7) and np.allclose(array(env.imported_inertias[0,i]).reshape(3,3),inertia,atol=2e-8),mass_kg=mass)
    expected=sum(float(l.find('inertial/mass').attrib['value']) for l in upstream.findall('link') if l.find('inertial/mass') is not None)
    check('import_total',np.isclose(float(env.imported_masses[0].sum()),expected,atol=1e-7),total_kg=expected)
    check('payload_total',np.isclose(float(env.nominal_masses[0].sum()),expected+env.physics['base_payload']['mass_kg'],atol=1e-7),total_kg=float(env.nominal_masses[0].sum()))
    # Nominal and partial resets must never re-add payload, nor disturb another env.
    with torch.no_grad():
        for _ in range(3):env.reset(seed=4242)
        check('no_payload_accumulation',torch.allclose(view.get_masses(),env.nominal_masses) and torch.allclose(view.get_inertias(),env.nominal_inertias,atol=1e-8))
        check('nominal_solver_effort_limit',np.allclose(array(view.get_dof_max_forces()),1.5))
        check('nominal_solver_velocity_limit',np.allclose(array(view.get_dof_max_velocities()),4.))
        raw=[]
        env.cfg.stage='robust'
        for seed in range(8):
            env.reset(seed=4242+seed)
            masses=view.get_masses().clone(); inertias=view.get_inertias().clone()
            ratio=masses/env.nominal_masses
            check('mass_inertia_scale_'+str(seed),bool(((ratio>=.8-1e-6)&(ratio<=1.2+1e-6)).all()) and torch.allclose(inertias,env.nominal_inertias*ratio[:,:,None],atol=1e-8))
            check('payload_COM_fixed_'+str(seed),torch.allclose(view.get_coms()[...,:3],env.nominal_coms[...,:3],atol=1e-7))
            effort=view.get_dof_max_forces(); velocity=view.get_dof_max_velocities()
            check('robust_solver_limits_'+str(seed),bool(((effort>=.98)&(effort<=2.94)&(velocity>=3.)&(velocity<=4.7)).all()))
            raw.append({'mass_scale':array(ratio[:,0]).tolist(),'effort_nm':array(effort[:,0]).tolist(),'velocity_rad_s':array(velocity[:,0]).tolist()})
            oldm=masses[1:].clone(); oldi=inertias[1:].clone(); olde=effort[1:].clone()
            env._reset_idx(torch.tensor([0],device=env.device))
            check('partial_randomization_isolated_'+str(seed),torch.equal(view.get_masses()[1:],oldm) and torch.equal(view.get_inertias()[1:],oldi) and torch.equal(view.get_dof_max_forces()[1:],olde))
        report['robust_samples']=raw
        env.cfg.stage='recover'
        zero=torch.zeros((64,6),device=env.device)
        for side in ('upright','left','right'):
            env.cfg.initial_side=side
            env.reset(seed=4242)
            fault=False
            for _ in range(round(20/env.step_dt)):
                obs,reward,term,trunc,_=env.step(zero)
                fault|=bool(term.any() or trunc.any())
                assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            pose=env.robot.data.root_link_pose_w[0].clone();pose[:3]-=env.scene.env_origins[0];pose[:2]=0
            pose[3:]/=torch.linalg.vector_norm(pose[3:])
            up=float(-env.robot.data.projected_gravity_b[0,2])
            check('settled_'+side,not fault and (up>.8 if side=='upright' else up<.2),upright_cos=up,max_root_speed=float(env.robot.data.root_vel_w.abs().max()))
            report['settled_starts'][side]={'root_pose':array(pose).tolist(),'joint_rad':array(env.robot.data.joint_pos[0,env.joint_ids]).tolist()}
            print('Settled',side,up,flush=True)
    report['passed']=all(c['passed'] for c in report['checks'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    config=json.loads(Path(args.physics).read_text())
    config['settled_starts']=report['settled_starts']
    config['side_upright_progress_reward_cap']=1.
    config['settled_start_provenance']={'report':str(args.output/'report.json'),'report_sha256':file_hash(args.output/'report.json'),'meaning':'Nominal physics-v2 rest after20s home hold, zero-velocity reset; xy recentered. Provisional mass/actuators, not physically measured.'}
    (args.output/'physics-v2-001.json').write_text(json.dumps(config,indent=2)+'\n')
    print('Physics-v2 import/settling audit complete',flush=True)
except BaseException:
    traceback.print_exc()
    import omni.kit.app
    omni.kit.app.get_app().post_quit(1)
    raise
finally:
    if env is not None:env.close()
    sys.stdout.flush();sys.stderr.flush();app.close()
