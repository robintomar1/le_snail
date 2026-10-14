import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from le_snail.assets import prepare_urdf
from le_snail.contract import JOINT_NAMES, PolicyContract


def source_model(path):
    root = ET.Element('robot', name='fixture')
    base = ET.SubElement(root, 'link', name='base_link')
    visual = ET.SubElement(base, 'visual')
    ET.SubElement(visual, 'origin', xyz='0 0 0.02')
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(geometry, 'box', size='0.1 0.1 0.04')
    parent = 'base_link'
    for name in JOINT_NAMES:
        child = name + '_link'
        ET.SubElement(root, 'link', name=child)
        joint = ET.SubElement(root, 'joint', name=name, type='revolute')
        ET.SubElement(joint, 'parent', link=parent)
        ET.SubElement(joint, 'child', link=child)
        ET.SubElement(joint, 'limit', lower='-3', upper='3', effort='10', velocity='10')
        parent = child
    ET.ElementTree(root).write(path)


def test_preparation_restores_missing_base_contact_preserving_transform(tmp_path):
    source = tmp_path/'source.urdf'
    source_model(source)
    contract = PolicyContract.load(Path(__file__).resolve().parents[1]/'configs/policy.json')
    output = tmp_path/'prepared.urdf'
    manifest = prepare_urdf(source, output, contract)
    base = ET.parse(output).find('link')
    assert base.find('collision/origin').attrib['xyz'] == '0 0 0.02'
    assert base.find('collision/geometry/box').attrib['size'] == '0.1 0.1 0.04'
    assert manifest['restored_collision_links'] == ['base_link']
    assert not manifest['contact_geometry_validated']
    assert ET.parse(source).find('link/collision') is None


def test_anchored_base_is_rejected(tmp_path):
    source = tmp_path/'source.urdf'
    source_model(source)
    tree = ET.parse(source)
    joint = ET.SubElement(tree.getroot(), 'joint', name='anchor', type='fixed')
    ET.SubElement(joint, 'parent', link='world')
    ET.SubElement(joint, 'child', link='base_link')
    tree.write(source)
    contract = PolicyContract.load(Path(__file__).resolve().parents[1]/'configs/policy.json')
    with pytest.raises(ValueError, match='unanchored'):
        prepare_urdf(source, tmp_path/'out.urdf', contract)
