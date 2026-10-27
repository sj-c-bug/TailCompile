"""Matched H x Z architectures for the pinned Hecate 3D implementation.

H changes typed hard-resource placement. Z changes only the coordinates of a
fixed number of otherwise identical scatter/gather instances. Actual RRG audits,
not this XML description, certify a capacity match.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LEGACY = REPO / 'experiments' / 'vtr3d'
COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'
SOURCE_ARCH = 'vtr_flow/arch/multi_die/hecate/3d/hecate_3d_sb_10um_bidir_fanin_16_fanout_16_with_esd_7nm.xml'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module(name):
    if not (LEGACY/(name+'.py')).is_file():
        raise FileNotFoundError(f'missing tracked 3D helper in the TailCompile repository: {name}')
    if str(LEGACY) not in sys.path:
        sys.path.insert(0, str(LEGACY))
    spec = importlib.util.spec_from_file_location('_v7_legacy_' + name, LEGACY / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vertical_locations(width, height, concentrated=False):
    """A geometry-only quarter-density grid versus an equally sized center patch.

    Locations are interior switch blocks. No netlist, placement or route is read.
    Each selected coordinate receives six identical bidirectional Z resources.
    D=100% denotes this frozen reference budget, not every possible location.
    """
    interior = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]
    spread = [(x, y) for x, y in interior if x % 2 == 1 and y % 2 == 1]
    if not spread:
        raise ValueError('grid has no interior vertical locations')
    if not concentrated:
        return spread
    cx, cy = (width - 1) / 2, (height - 1) / 2
    selected = sorted(interior, key=lambda p: (max(abs(p[0]-cx), abs(p[1]-cy)),
                                             (p[0]-cx)**2 + (p[1]-cy)**2, p))[:len(spread)]
    if set(selected) == set(spread):
        raise ValueError('grid too small to distinguish spread and concentrated Z locations')
    return sorted(selected)


def generate_architecture(source_arch, out, h=0, z=0, width=16, height=16,
                          hard_columns=4, z_per_location=6):
    if h not in (0, 1) or z not in (0, 1):
        raise ValueError('H and Z must be 0 or 1')
    if not isinstance(z_per_location, int) or z_per_location < 1:
        raise ValueError('z_per_location must be a positive integer')
    source_arch, out = Path(source_arch), Path(out)
    metadata = _module('pilot_architectures').generate_architecture(
        source_arch, out, 'heterogeneous' if h else 'homogeneous', width, height, hard_columns)
    tree = ET.parse(out)
    root = tree.getroot()
    patterns = root.findall('./scatter_gather_list/sg_pattern')
    if len(patterns) != 1:
        raise ValueError('v7 adapter requires one audited Hecate scatter/gather pattern')
    pattern = patterns[0]
    if pattern.get('type') != 'bidir':
        raise ValueError('expected the pinned bidirectional scatter/gather pattern')
    up = pattern.find("./sg_link_list/sg_link[@name='L_UP']")
    if up is None or up.get('z_offset') != '1':
        raise ValueError('expected the pinned adjacent-layer L_UP link')
    for location in list(pattern.findall('sg_location')):
        pattern.remove(location)
    locations = vertical_locations(width, height, bool(z))
    for x, y in locations:
        ET.SubElement(pattern, 'sg_location', type='XY_SPECIFIED',
                      num=str(z_per_location), x=str(x), y=str(y), sg_link_name='L_UP')
    ET.indent(tree, space='  ')
    tree.write(out, encoding='utf-8', xml_declaration=True)
    source = ET.parse(source_arch).getroot()
    def immutable(tree_root):
        return [(item.tag, ET.tostring(item, encoding='unicode')) for item in tree_root
                if item.tag not in ('layout', 'fixed_layout', 'scatter_gather_list')]
    # Formatting changes are ignored; XML values and primitive structure are not.
    def canonical(node):
        return (node.tag, sorted(node.attrib.items()), (node.text or '').strip(),
                [canonical(child) for child in node])
    protected = lambda r: [canonical(n) for n in r if n.tag not in ('layout', 'fixed_layout', 'scatter_gather_list')]
    if protected(source) != protected(root):
        raise AssertionError('primitive, pin, switch or segment definitions changed')
    metadata.update(schema='tailcompile-v7-hz-architecture-1', condition=f'H{h}Z{z}', H=h, Z=z,
                    architecture_sha256=sha(out), vertical_routing_definitions_unchanged=False,
                    primitive_switch_segment_definitions_unchanged=True,
                    z_locations=[list(p) for p in locations], z_location_count=len(locations),
                    z_per_location=z_per_location,
                    declared_z_instance_count=len(locations)*z_per_location,
                    z_budget_semantics='reference selected-location budget; actual RRG capacity must be audited',
                    z_selection='center Chebyshev patch' if z else 'odd-coordinate regular grid',
                    z_geometry_only=True,
                    channel_width_z_coupling='unverified until actual per-width RRG audit')
    return metadata


def audit_family(graphs):
    """Verify actual total resources/Z classes at each width and report coupling.

    graphs is a mapping condition -> list of device_graph JSON dictionaries.
    Missing conditions are an error rather than a silently incomplete match.
    """
    required = {'H0Z0', 'H0Z1', 'H1Z0', 'H1Z1'}
    if set(graphs) != required:
        raise ValueError('audit needs all four H/Z conditions')
    by_width = {}
    for condition, values in graphs.items():
        for graph in values:
            width = int(graph['channel_width'])
            entry = by_width.setdefault(width, {})
            if condition in entry:
                raise ValueError('duplicate condition/width')
            entry[condition] = graph['audit']
    for width, rows in by_width.items():
        if set(rows) != required:
            raise ValueError(f'width {width} has an incomplete four-cell audit')
        for field in ('grid', 'total_resource_capacity', 'z_capacity_by_class', 'z_switch_classes',
                      'z_actual_adjacent_switch_classes'):
            values = [r[field] for r in rows.values()]
            if any(value != values[0] for value in values[1:]):
                raise ValueError(f'width {width}: four cells do not match {field}: {values}')
        for h in (0, 1):
            spread, center = rows[f'H{h}Z0'], rows[f'H{h}Z1']
            if center['z_mean_squared_distance_to_center'] >= spread['z_mean_squared_distance_to_center']:
                raise ValueError(f'width {width}: actual Z1 locations are not more concentrated')
        for z in (0, 1):
            if rows[f'H0Z{z}']['z_capacity_by_xy'] != rows[f'H1Z{z}']['z_capacity_by_xy']:
                raise ValueError(f'width {width}: H changes actual Z location capacity')
    widths = sorted(by_width)
    coupling = {}
    for condition in sorted(required):
        values = {str(w): by_width[w][condition]['z_capacity_by_class'] for w in widths}
        reference = next(iter(values.values()))
        coupling[condition] = {'z_capacity_by_width': values,
                               'z_capacity_invariant': all(v == reference for v in values.values())}
    return dict(schema='tailcompile-v7-hz-actual-audit-1', four_cell_matches=True,
                channel_widths=widths, by_width=by_width, width_coupling=coupling,
                claim='actual independent connected CHANZ capacity/classes matched, Z1 more concentrated, H leaves Z positions unchanged',
                limitation='equal supply does not imply equal pin access or detailed routability')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-arch', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--h', type=int, choices=(0, 1), required=True)
    p.add_argument('--z', type=int, choices=(0, 1), required=True)
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--height', type=int, default=16)
    p.add_argument('--hard-columns', type=int, default=4)
    args = p.parse_args()
    value = generate_architecture(args.source_arch, args.out, args.h, args.z,
                                  args.width, args.height, args.hard_columns)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
