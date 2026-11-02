"""011 sparse edit observations, before native traces are pruned.

This describes what happened to edited atoms. Retention is a mechanism measure,
not a QoR objective, and absent/partial exports never imply zero movement.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _rows(path, gp):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.strip().split('\t') if gp else line.split()
            if gp:
                if len(fields) != 10:
                    raise ValueError('011 observation requires native ten-column GP export')
                name, x, y, z = fields[1], fields[4], fields[5], fields[6]
            else:
                if len(fields) < 5:
                    raise ValueError('011 observation requires atom x y layer subtile')
                name, x, y, z = fields[:4]
            xyz = tuple(map(float, (x, y, z)))
            if not all(math.isfinite(v) and v >= 0 for v in xyz):
                raise ValueError('invalid native coordinates in residual observation')
            yield name, xyz[0], xyz[1], math.floor(xyz[2]+.5)


def collect_residual_observations(generated, target, workdir):
    if not generated or not generated.get('residual_edits'):
        return dict(status='not_applicable', reason='no_residual_edit_manifest')
    path = Path(generated['residual_edits'])
    if _sha(path) != generated.get('residual_edits_sha256'):
        raise ValueError('residual edit manifest changed after initialization')
    edits = json.loads(path.read_text(encoding='utf-8'))
    width, height = edits['grid'][:2]
    if min(width, height) <= 0:
        raise ValueError('positive native XY normalization grid required')
    atoms = {r['name']: r for r in edits['atoms']}
    if len(atoms) != len(edits['atoms']):
        raise ValueError('duplicate edited atom')
    result = dict(schema='tailcompile-v7-residual-observations-011-1',
                  source_sha256=_sha(path), edited_atoms=len(atoms), stages={},
                  retention_is_not_a_qor_objective=True)
    targets = (('post_gp', Path(str(Path(target)/'post_gp')+'.blocks.tsv'), True),
               ('post_appack_legalization', Path(workdir)/'post_legalizer.fplace', False),
               ('final', Path(workdir)/'post_route.fplace', False))
    for stage, source, gp in targets:
        if not atoms:
            result['stages'][stage] = dict(status='no_edits', matched_atoms=0,
                                          layer_target_retained_fraction=None)
            continue
        if not source.is_file():
            result['stages'][stage] = dict(status='missing', matched_atoms=None,
                                          layer_target_retained_fraction=None)
            continue
        seen = set(); layer_edited = retained = reverted = 0
        distance_target = distance_base = 0.
        for name, x, y, layer in _rows(source, gp):
            if name not in atoms:
                continue
            if name in seen:
                raise ValueError('duplicate edited atom in native export')
            seen.add(name); row = atoms[name]
            if row['base_layer'] != row['target_layer']:
                layer_edited += 1
                retained += layer == row['target_layer']
                reverted += layer == row['base_layer']
            xy = (x/width, y/height)
            distance_target += sum(abs(a-b) for a, b in zip(xy, row['target_xy']))
            distance_base += sum(abs(a-b) for a, b in zip(xy, row['base_xy']))
        result['stages'][stage] = dict(status='observed' if len(seen)==len(atoms) else 'partial_coverage',
            source_sha256=_sha(source), matched_atoms=len(seen), missing_atoms=len(atoms)-len(seen),
            matched_fraction=len(seen)/len(atoms), matched_layer_edited_atoms=layer_edited,
            layer_target_retained_fraction=retained/layer_edited if layer_edited else None,
            layer_reverted_to_base_fraction=reverted/layer_edited if layer_edited else None,
            mean_normalized_xy_l1_to_target=distance_target/len(seen) if seen else None,
            mean_normalized_xy_l1_to_base=distance_base/len(seen) if seen else None,
            denominator='matched edited atoms; layer fractions use only actually layer-edited atoms')
    return result
