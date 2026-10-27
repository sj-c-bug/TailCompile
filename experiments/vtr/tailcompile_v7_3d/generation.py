"""Atom-level lowering of a whole-molecule 3D AP initializer."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

try:
    from .repository_guard import validate_output_roots
except ImportError:
    from repository_guard import validate_output_roots


def lower_initialization(logical, device, prediction, output, *, method=None):
    """Write native AP coordinates; physical site/BEL legality is not claimed.

    logical.clusters carry atoms and optional fixed_location=[x,y,z,subtile].
    A partially fixed location retains every known axis exactly. Output covers
    all atoms, including fixed atoms, so backend coverage can be audited.
    """
    started = time.perf_counter()
    output = Path(output)
    validate_output_roots((output,))
    output.mkdir(parents=True, exist_ok=True)
    grid = device.get('grid', {})
    width = device.get('grid_width', grid.get('width'))
    height = device.get('grid_height', grid.get('height'))
    layers = device.get('layer_count', device.get('layers', grid.get('layers')))
    if not width or not height or not layers:
        raise ValueError('explicit 3D grid required for native lowering')
    def values(value):
        return value.detach().cpu().tolist() if hasattr(value, 'detach') else value
    xy, layer = values(prediction['xy']), values(prediction['layer'])
    clusters = logical['clusters']
    if len(xy) != len(clusters) or len(layer) != len(clusters):
        raise ValueError('prediction/APBlock coverage mismatch')
    rows, atoms = [], set()
    for node, block in enumerate(clusters):
        x, y = xy[node]
        z = int(layer[node])
        if not all(math.isfinite(v) and 0 <= v < 1 for v in (x, y)) or z != layer[node] or not 0 <= z < layers:
            raise ValueError(f'invalid normalized XYZ for molecule {node}')
        native = [x * width, y * height, z, 0]
        fixed = block.get('fixed_location')
        if fixed is not None:
            for axis in range(4):
                if fixed[axis] >= 0:
                    if axis == 2 and z != fixed[axis]:
                        raise ValueError('prediction moved a known fixed layer')
                    if axis < 2 and not math.isclose(native[axis], fixed[axis], rel_tol=0., abs_tol=2e-5):
                        raise ValueError('prediction moved a known fixed XY coordinate')
                    native[axis] = fixed[axis]
        if not block.get('atoms'):
            raise ValueError('APBlock has no atoms')
        for atom in block['atoms']:
            if not isinstance(atom, str) or not atom or atom.startswith('#') or any(c.isspace() for c in atom):
                raise ValueError('atom name is not TSV-safe')
            if atom in atoms:
                raise ValueError(f'duplicate atom across molecules: {atom}')
            atoms.add(atom)
            rows.append(f'{atom}\t{native[0]:.17g}\t{native[1]:.17g}\t{int(native[2])}\t{int(native[3])}\n')
    path = output / 'init.ap.tsv'
    content = '# tailcompile-v7-3d native AP initialization; native GP must continue\n' + ''.join(rows)
    path.write_text(content, encoding='utf-8')
    target = dict(assignment=values(prediction['assignment']), xy=xy, layer=layer)
    target_path = output / 'target.json'
    target_path.write_text(json.dumps(target, indent=2), encoding='utf-8')
    manifest = dict(schema='tailcompile-v7-3d-initialization-1',
        logical_unit='apblock', initializer=method or prediction.get('initializer', 'unknown'),
        backend='native_gp_appack', atom_count=len(atoms), molecule_count=len(clusters),
        initial_layer_counts={str(z): layer.count(z) for z in sorted(set(layer))},
        ap_init=str(path.resolve()), ap_init_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        target=str(target_path.resolve()), target_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),
        generation_seconds=time.perf_counter() - started,
        decoder_kind=prediction.get('decoder_kind'), capacity_diagnostics=prediction.get('capacity_diagnostics'),
        phase_seconds=prediction.get('phase_seconds'), native_gp_required=True,
        packing_legality_claimed=False)
    (output / 'generation.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
