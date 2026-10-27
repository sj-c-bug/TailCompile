"""Finite real XYZ initialization -> native GP -> APPack -> route qualification."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from architectures import SOURCE_ARCH, _module
from data import read_snapshot
from io_utils import (COMMIT, emit, freeze, read, run_trial, sha, vpr_command, write,
                      install_signal_handlers, raise_if_cancelled)


def write_prior(snapshot, path, *, flip=False):
    from supervision import native_layer
    W, H, layers = snapshot['size']
    lines = ['# qualification only; intact AP blocks; native GP must continue\n']
    fixed, counts = 0, Counter()
    for block in snapshot['blocks']:
        x, y, z, sub = block['loc']
        z = native_layer(z, layers)
        if flip and not block['fixed']:
            z = layers - 1 - z
        fixed += int(block['fixed'])
        counts[str(z)] += 1
        if not 0 <= x < W or not 0 <= y < H:
            raise ValueError('teacher XY outside native AP grid')
        for atom in block['atoms']:
            lines.append(f'{atom}\t{x:.17g}\t{y:.17g}\t{z}\t{max(0, sub)}\n')
    path = Path(path)
    path.write_text(''.join(lines), encoding='utf-8')
    return dict(path=str(path), sha256=sha(path), fixed_blocks=fixed, layer_blocks=dict(counts))


def flat_summary(path):
    path = Path(path)
    if not path.exists():
        return dict(present=False)
    layers, atoms, coords = Counter(), 0, {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        row = line.split()
        if len(row) < 5:
            continue
        try:
            x, y, z, sub = map(float, row[1:5])
        except ValueError:
            continue
        layers[str(int(z))] += 1
        atoms += 1
        coords[row[0]] = [x, y, z, sub]
    return dict(present=True, atom_count=atoms, layer_atom_counts=dict(layers),
                sha256=sha(path), positions=coords)


def make_fixed_constraints(out, blif, atom):
    """One true single-location floorplan, not a learned hard layer prior."""
    root = ET.Element('vpr_constraints', tool_name='vpr')
    partitions = ET.SubElement(root, 'partition_list')
    partition = ET.SubElement(partitions, 'partition', name='qualification_fixed_lut')
    ET.SubElement(partition, 'add_atom', name_pattern=atom)
    ET.SubElement(partition, 'add_region', x_low='1', y_low='1', x_high='1', y_high='1',
                  layer_low='1', layer_high='1')
    ET.indent(root)
    ET.ElementTree(root).write(out, encoding='utf-8', xml_declaration=True)


def prepare_cases(args):
    fixtures = _module('heterogeneity_fixtures')
    source = args.vtr_root / SOURCE_ARCH
    cases = []
    for name, fixture in (('aligned', 'aligned'), ('shifted', 'shifted'), ('fixed_mixed', 'asymmetric')):
        folder = args.out/'inputs'/name
        if not (folder/'fixture.json').exists():
            fixtures.generate_case(source, folder, fixture)
        cases.append(dict(name=name, arch=folder/'arch.xml', blif=folder/'tiny.blif', width=320))
    folder = args.out/'inputs'/'sha'
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder/'arch.xml').exists():
        _module('pilot_architectures').generate_architecture(source, folder/'arch.xml', 'heterogeneous', 16, 16)
    if not (folder/'design.blif').exists():
        adapted = _module('prepare_natural_blif').prepare_blif(args.sha_blif, folder/'design.blif', folder/'arch.xml')
        write(folder/'mapping.json', adapted)
    cases.append(dict(name='sha', arch=folder/'arch.xml', blif=folder/'design.blif', width=320))
    for case in cases:
        if case['name'] == 'fixed_mixed':
            # The generated fixture names the first reduction LUT lane0_reduce_0;
            # select an actual .names output instead of guessing source naming.
            atom = next(line.split()[-1] for line in case['blif'].read_text().splitlines()
                        if line.startswith('.names ') and len(line.split()) >= 3)
            constraint = case['arch'].parent/'fixed.xml'
            if not constraint.exists():
                make_fixed_constraints(constraint, case['blif'], atom)
            case['constraints'] = constraint
    return cases


def condition(args, case, seed):
    base = args.out/'runs'/case['name']/f'seed{seed}'
    exports = base/'exports'
    exports.mkdir(parents=True, exist_ok=True)
    raw = exports/'gp_default'
    trace = exports/'structure'
    command = vpr_command(args.binary, case['arch'], case['blif'], case['width'], seed)
    inputs = [case['arch'], case['blif']]
    if case.get('constraints'):
        command += ['--read_vpr_constraints', str(case['constraints'])]
        inputs.append(case['constraints'])
    defaults = run_trial(base/'default', command + [
        '--write_legalized_flat_place', str(exports/'default_legalized.fplace'),
        '--write_flat_place', str(exports/'default_post_sa.fplace')],
        inputs=inputs, env=dict(TC_GP_EXPORT=raw, TC_GP_STRUCTURE=trace), timeout=args.timeout,
        metadata=dict(stage='P1_qualification', case=case['name'], seed=seed,
                      binary_sha256=args.binary_hash, method='default'))
    if defaults['status'] != 'success':
        return dict(case=case['name'], seed=seed, default=defaults,
                    passed=False, blocker='native_default_did_not_complete')
    snapshot = read_snapshot(raw)
    structure = read_snapshot(trace, structure=True)
    if snapshot['size'] != structure['size']:
        raise ValueError('Default/structure grid mismatch')
    rows = [dict(method='default', trial=defaults,
                 post_legalizer=flat_summary(exports/'default_legalized.fplace'),
                 post_sa=flat_summary(exports/'default_post_sa.fplace'))]
    prior_infos = []
    for method, flip in (('replay', False), ('changed_layer', True)):
        prior = exports/(method+'.ap.tsv')
        info = write_prior(snapshot, prior, flip=flip)
        prior_infos.append(info)
        injected = exports/(method+'_injected')
        gp = exports/(method+'_gp')
        trial = run_trial(base/method, command + [
            '--write_legalized_flat_place', str(exports/(method+'_legalized.fplace')),
            '--write_flat_place', str(exports/(method+'_post_sa.fplace'))], inputs=inputs+[prior],
            env=dict(TC_GP_INIT=prior, TC_GP_TRACE=injected, TC_GP_EXPORT=gp), timeout=args.timeout,
            metadata=dict(stage='P1_qualification', case=case['name'], seed=seed,
                          binary_sha256=args.binary_hash, method=method))
        text = Path(trial['log']).read_text(errors='replace')
        evidence = dict(consumed='TAILCOMPILE_V7_GP_INIT_CONSUMED' in text,
                        native_continued='TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' in text,
                        appack_grid_logged='TAILCOMPILE_V7_POST_APPACK' in text,
                        did_not_skip_gp='skipping the Global Placement' not in text)
        injected_snapshot = read_snapshot(injected) if Path(str(injected)+'.meta.tsv').exists() else None
        if injected_snapshot:
            counts = Counter(str(int(b['loc'][2])) for b in injected_snapshot['blocks'])
            evidence['injected_layer_counts'] = dict(counts)
            evidence['nonzero_layer_injected'] = any(int(z) > 0 and count for z, count in counts.items())
            evidence['matches_requested_layer_counts'] = dict(counts) == info['layer_blocks']
        rows.append(dict(method=method, trial=trial, evidence=evidence,
                         post_gp_layer=Counter(str(b['loc'][2]) for b in read_snapshot(gp)['blocks'])
                            if Path(str(gp)+'.meta.tsv').exists() else {},
                         post_legalizer=flat_summary(exports/(method+'_legalized.fplace')),
                         post_sa=flat_summary(exports/(method+'_post_sa.fplace'))))
    checks = dict(all_routed=all(row['trial']['status']=='success' for row in rows),
        warm_consumed_and_continued=all(row['evidence']['consumed'] and row['evidence']['native_continued']
                                       and row['evidence']['did_not_skip_gp'] for row in rows[1:]),
        requested_layer_counts_preserved=all(row['evidence'].get('matches_requested_layer_counts', False)
                                            for row in rows[1:]),
        nonzero_layer_covered=any(row['evidence'].get('nonzero_layer_injected', False) for row in rows[1:]),
        real_chanz=any((row['trial']['metrics']['unique_chanz_nodes'] or 0)>0 for row in rows),
        fixed_fixture_has_fixed_apblock=any(b['fixed'] for b in snapshot['blocks'])
             if case['name']=='fixed_mixed' else True,
        atom_coverage=all(row['post_legalizer'].get('atom_count',0)>0 and
                          row['post_sa'].get('atom_count',0)>0 for row in rows))
    report = dict(case=case['name'], seed=seed, grid=snapshot['size'],
                  checks=checks, passed=all(checks.values()), rows=rows, priors=prior_infos)
    write(base/'qualification.json', report)
    return report


def main():
    install_signal_handlers()
    raise_if_cancelled()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--vtr-root', type=Path, required=True)
    parser.add_argument('--sha-blif', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--cases', nargs='+', default=['aligned','shifted','fixed_mixed','sha'])
    args = parser.parse_args()
    for key in ('binary','vtr_root','sha_blif','out'):
        setattr(args, key, getattr(args,key).resolve())
    from repository_guard import validate_output_roots
    validate_output_roots((args.out,),dependency_roots=(args.vtr_root,))
    args.out.mkdir(parents=True, exist_ok=True)
    previous = args.out/'qualification_report.json'
    if previous.exists() and read(previous).get('harness_version', 1) == 1:
        write(args.out/'qualification_report.harness_v1.json', read(previous))
    args.binary_hash = sha(args.binary)
    freeze(args.out/'protocol.json', dict(schema='tailcompile-v7-p1-1', commit=COMMIT,
        binary_sha256=args.binary_hash, seeds=[701,702], methods=['default','replay','changed_layer'],
        cases=args.cases, timeout=args.timeout, initializer='engineering_probe_not_gnn',
        performance_selection=False))
    cases = [case for case in prepare_cases(args) if case['name'] in args.cases]
    tasks = [(case,seed) for case in cases for seed in (701,702)]
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {}
        for case,seed in tasks:
            raise_if_cancelled()
            futures[pool.submit(condition,args,case,seed)] = (case['name'],seed)
        for future in as_completed(futures):
            name,seed = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = dict(case=name,seed=seed,passed=False,error=f'{type(exc).__name__}: {exc}')
            results.append(result)
            write(args.out/'qualification_report.json',dict(schema='tailcompile-v7-p1-1',
                harness_version=2,
                harness_note='A replay entirely on layer 1 flips entirely to layer 0. Verify the requested distribution, rather than incorrectly requiring each flipped prior to retain nonzero Z.',
                complete=len(results)==len(tasks), passed=len(results)==len(tasks) and all(r['passed'] for r in results),
                condition_count=len(tasks), completed_conditions=len(results), conditions=results))
            emit('qualification_condition',case=name,seed=seed,passed=result['passed'])
    report = read(args.out/'qualification_report.json')
    print(dict(complete=report['complete'],passed=report['passed'],conditions=len(results)))
    raise SystemExit(0 if report['passed'] else 1)


if __name__=='__main__':
    main()
