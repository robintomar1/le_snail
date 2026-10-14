"""Validate and prepare the designer's URDF without inventing robot geometry."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .contract import JOINT_NAMES, PolicyContract


def prepare_urdf(source, destination, contract: PolicyContract):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    root = ET.parse(source).getroot()
    links = {node.attrib['name']: node for node in root.findall('link')}
    if 'base_link' not in links:
        raise ValueError('Expected SO-101 base_link')
    movable = {j.attrib['name']: j for j in root.findall('joint') if j.attrib.get('type') != 'fixed'}
    if set(movable) != set(JOINT_NAMES):
        raise ValueError(f'Unexpected joints: {sorted(movable)}')
    fixed_parents = {j.find('child').attrib['link'] for j in root.findall('joint')}
    if 'base_link' in fixed_parents:
        raise ValueError('base_link must be the unanchored root')
    restored = []
    for name, link in links.items():
        # Restore only missing collisions. Preserve all existing geometry and transforms.
        if not link.findall('collision') and link.findall('visual'):
            for visual in link.findall('visual'):
                collision = ET.SubElement(link, 'collision')
                for field in ('origin', 'geometry'):
                    node = visual.find(field)
                    if node is not None:
                        collision.append(deepcopy(node))
            restored.append(name)
    if not links['base_link'].findall('collision'):
        raise ValueError('Base has neither collision geometry nor usable visuals')
    meshes = {}
    for mesh in root.findall('.//mesh'):
        name = mesh.attrib['filename']
        if '://' in name or Path(name).is_absolute():
            raise ValueError(f'Expected repository-relative mesh: {name}')
        path = (source.parent / name).resolve()
        if not path.is_relative_to(source.parent) or not path.is_file():
            raise ValueError(f'Missing or external mesh: {name}')
        meshes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        mesh.set('filename', str(path))
    limits = {}
    for i, name in enumerate(JOINT_NAMES):
        limit = movable[name].find('limit')
        lo, hi = float(limit.attrib['lower']), float(limit.attrib['upper'])
        if contract.lower_rad[i] < lo or contract.upper_rad[i] > hi:
            raise ValueError(f'Policy limits for {name} exceed the source URDF')
        limits[name] = {'lower': lo, 'upper': hi}
        # Imported effort/velocity values are placeholders; runtime actuator config overrides them.
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
    manifest = {
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'prepared_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'restored_collision_links': restored, 'mesh_sha256': meshes,
        'joint_limits': limits, 'contract_fingerprint': contract.fingerprint,
        'contact_geometry_validated': False,
    }
    destination.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
