#!/usr/bin/env python3
"""CPU-only schema, timing and observable-command conflict gate for teacher data."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
if args.output.exists():p.error('--output must be new')
args.output.mkdir(parents=True)
started=time.monotonic()
report=json.loads((args.input/'report.json').read_text())
assert report.get('collection_complete') and report.get('positive_control_passed')
contract=json.loads(Path('configs/policy.json').read_text())
scale=np.asarray(contract['action_scale_rad']); home=np.asarray(contract['home_rad'])
lower=np.asarray(contract['lower_rad']); upper=np.asarray(contract['upper_rad']); slew=np.asarray(contract['max_target_speed_rad_s'])*report['policy_dt']
# Physical units are preregistered; convert normalized applied-action history to radians.
tolerance=np.array([.01]*6+[.05]*6+[.01]*3+[.05]*3+(.01/scale).tolist())
observations=[];actions=[];applied=[];metadata=[];counts={};checks=[]
for group in report['groups']:
 data=np.load(args.input/f"{group['stem']}.npz")
 obs,act,pre,post=data['observations'],data['desired_action'],data['pre_target'],data['post_applied']
 assert obs.shape[2]==24 and act.shape[2]==6
 assert all(np.isfinite(data[k]).all() for k in data.files)
 desired=np.clip(home+np.clip(act,-1,1)*scale,lower,upper)
 expected=pre+np.clip(desired-pre,-slew,slew)
 error=float(np.abs(expected-post).max())
 assert error<2e-6,error
 assert np.max(np.abs(obs[:,:,:6]-(data['joint_q']-home)))<2e-6
 assert np.max(np.abs(obs[:,:,6:12]-data['filtered_velocity']))<2e-6
 assert np.max(np.abs(obs[:,:,18:]-(data['pre_applied']-home)/scale))<2e-6
 for trial in group['trials']:
  if not trial['success']:continue
  env=trial['env'];last=round(trial['acceptance_end_s']/report['policy_dt'])
  indices=np.flatnonzero(data['active'][:last,env])
  variant=group['variants'][env]
  observations.append(obs[indices,env]);actions.append(act[indices,env]);applied.append(post[indices,env])
  metadata.extend([{'group':group['stem'],'side':group['side'],'repeat':group['repeat'],'env':env,'variant':variant,'step':int(step),'phase':int(data['phase'][step,env])}for step in indices])
 counts[group['stem']]={'successful_episodes':group['successes'],'timing_max_error_rad':error}
assert observations,'No successful demonstration episodes'
obs=np.concatenate(observations);act=np.concatenate(actions);command=np.concatenate(applied)
# Float64 tree avoids silently relaxing physical-unit neighborhoods.
features=obs.astype(np.float64)/tolerance
tree=cKDTree(features)
conflicts=[];conflicting_pairs=0;near_pairs=0;worst=0.;affected=np.zeros(len(obs),dtype=bool)
# Bound memory by batches; include all side and upright positive-prefix rows, not only a sampled training subset.
for start in range(0,len(obs),128):
 neighborhoods=tree.query_ball_point(features[start:start+128],r=1.,p=np.inf,workers=1)
 for offset,neighbors in enumerate(neighborhoods):
  i=start+offset;neighbors=np.asarray([j for j in neighbors if j>i],dtype=np.int64)
  if not len(neighbors):continue
  near_pairs+=len(neighbors)
  differences=np.max(np.abs(command[neighbors]-command[i]),axis=1)
  worst=max(worst,float(differences.max()))
  bad=neighbors[differences>.025]
  if len(bad):
   conflicting_pairs+=len(bad);affected[i]=True;affected[bad]=True
   for j in bad[:2]:
    if len(conflicts)<100:
     conflicts.append({'row_a':int(i),'row_b':int(j),'a':metadata[i],'b':metadata[j],
                       'max_next_applied_difference_rad':float(np.abs(command[j]-command[i]).max()),
                       'observation_difference':(obs[j]-obs[i]).tolist(),
                       'normalized_neighborhood_distance':float(np.abs(features[j]-features[i]).max()),
                       'command_a':command[i].tolist(),'command_b':command[j].tolist(),
                       'desired_action_a':act[i].tolist(),'desired_action_b':act[j].tolist()})
 print(json.dumps({'audited_rows':min(start+128,len(obs)),'total_rows':len(obs),'conflicting_pairs':conflicting_pairs}),flush=True) if start%4096==0 else None

result={'schema_version':1,'input':str(args.input),'input_report_sha256':hashlib.sha256((args.input/'report.json').read_bytes()).hexdigest(),
        'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),Path('scripts/collect_recovery_demonstrations.py'),Path('configs/policy.json'),Path('configs/experiments/physics-v2-002.json')]},
        'raw_file_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in args.input.glob('*.npz')},
        'counts':counts,'positive_prefix_rows':len(obs),'near_pairs':near_pairs,'conflicting_pairs':conflicting_pairs,'affected_rows':int(affected.sum()),
        'max_near_next_applied_difference_rad':worst,'ambiguity_criteria':report['ambiguity_preregistered'],
        'conflict_examples':conflicts,'dataset_valid':conflicting_pairs==0,
        'interpretation':'Unexplained material next-applied-command contradictions fail the preregistered gate. Desired actions outside clipping may alias; those alone are not conflicts. No observed collision would not prove global feedforward representability. Phase stays metadata, never actor input.',
        'elapsed_cpu_seconds':time.monotonic()-started}
if not conflicting_pairs:
 # Versioned, equal-sized side/upright data, with equal counts per script variant.
 rng=np.random.default_rng(10142)
 ids={v:np.array([i for i,m in enumerate(metadata)if m['variant']==v])for v in (9,34,48,62)}
 upright=np.array([i for i,m in enumerate(metadata)if m['side']=='upright'])
 assert all(len(v)>0 for v in ids.values()) and len(upright)>0
 per_variant=min(min(map(len,ids.values())),len(upright)//4)
 assert per_variant>0
 side_ids=np.concatenate([rng.choice(ids[v],per_variant,replace=False)for v in (9,34,48,62)])
 up_ids=rng.choice(upright,len(side_ids),replace=False)
 np.savez_compressed(args.output/'dataset.npz',side_observations=obs[side_ids],side_actions=act[side_ids],side_variant=np.array([metadata[i]['variant']for i in side_ids]),upright_observations=obs[up_ids],upright_actions=act[up_ids])
 result['training_rows_per_set']=len(side_ids)
 result['dataset_sha256']=hashlib.sha256((args.output/'dataset.npz').read_bytes()).hexdigest()
(args.output/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ('conflict_examples','raw_file_sha256','counts')},indent=2),flush=True)
# A failed data gate is a preserved scientific result; nonzero alerts orchestration.
raise SystemExit(0 if result['dataset_valid'] else 2)
