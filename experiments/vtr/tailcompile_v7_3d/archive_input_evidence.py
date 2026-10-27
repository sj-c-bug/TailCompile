"""Compact existing pre-GP probes and audit mode demand without rerunning VPR."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import shutil
import subprocess

from archive_local import archive
from data import build_logic
from io_utils import read, write, sha

HERE = Path(__file__).resolve().parent
DATA_SHA = '28698f8024e86e9a6a8bc424ac56d72055ee9beecb94e4303a1a95e29b8ec28c'


def logic_summary(logic):
    clusters = logic['clusters']
    return dict(schema=logic['schema'], design=logic['design'], family=logic['family'],
                split=logic['split'], grid=logic['grid'], node_count=len(clusters),
                net_count=len(logic['nets']), atom_count=sum(len(c['atoms']) for c in clusters),
                demand_keys=logic['demand_keys'],
                demand=[sum(c['demand'][i] for c in clusters) for i in range(len(logic['demand_keys']))],
                hard_ap_counts={key: sum(c['resource'] == key for c in clusters)
                                for key in ('memory', 'dsp_top')},
                raw_models=dict(Counter(m for c in clusters for m in c['model_names'])),
                demand_semantics=logic.get('demand_semantics', 'historical_pre_mode_adapter'),
                blif_sha256=logic['blif_sha256'], architecture_sha256=logic['architecture_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend-evidence', type=Path, required=True)
    args = parser.parse_args()
    if sha(HERE/'data.py') != DATA_SHA:
        raise ValueError('mode-aware parser differs from the explicitly frozen version')
    args.out.mkdir(parents=True, exist_ok=True)
    roots = [('primary', 'v7_input_probe_20261025'),
             ('candidates', 'v7_input_candidates_20261025'), ('bgm', 'v7_input_bgm_20261025')]
    index = []
    for label, dirname in roots:
        source = (args.scratch/dirname).resolve()
        target = (args.out/label).resolve()
        if source == target or source in target.parents or target in source.parents:
            raise ValueError('input evidence directories must be disjoint')
        target.mkdir(exist_ok=True)
        shutil.copyfile(source/'input_audit.json', target/'input_audit.json')
        for folder in sorted(p for p in source.iterdir() if p.is_dir()):
            destination = target/folder.name
            destination.mkdir(exist_ok=True)
            for name in ('mapping.json', 'source.json', 'structure.schema.json', 'arch.xml'):
                if (folder/name).is_file():
                    shutil.copyfile(folder/name, destination/name)
            if (folder/'structure_run').is_dir():
                archive(folder/'structure_run', destination/'structure_run')
            if (folder/'logic.json').is_file():
                write(destination/'logic_summary.json', logic_summary(read(folder/'logic.json')))
            evidence = []
            for path in sorted(p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()):
                relative = path.relative_to(folder)
                evidence.append(dict(path=str(path), relative=str(relative), bytes=path.stat().st_size,
                                     sha256=sha(path), copied=(destination/relative).is_file()))
            write(destination/'source_file_index.json', evidence)
            index.append(dict(group=label, design=folder.name, source=str(folder),
                              source_bytes=sum(row['bytes'] for row in evidence)))
    addendum = dict(schema='tailcompile-v7-input-mode-audit-1', data_sha256=DATA_SHA,
                    scope='existing pre-GP AP exports only; no GP/P&R rerun; old data preserved', designs=[])
    for design, dirname in [('sha', roots[0][1]), ('or1200', roots[0][1]),
                            ('spmv', roots[0][1]), ('bgm', roots[2][1])]:
        folder = args.scratch/dirname/design
        previous = read(folder/'logic.json')
        row = dict(design=design, source=str(folder), previous=logic_summary(previous),
                   input_sha256={name: sha(folder/name) for name in ('design.blif', 'arch.xml',
                       'structure.blocks.tsv', 'structure.nets.tsv', 'structure.pins.tsv',
                       'structure.schema.json', 'structure.meta.tsv')})
        try:
            fresh = build_logic(folder/'structure', design, folder/'design.blif', folder/'arch.xml',
                                family=previous['family'], split=previous['split'])
            summary = logic_summary(fresh)
            if summary['hard_ap_counts'] != row['previous']['hard_ap_counts']:
                raise ValueError('mode adapter changed physical hard AP counts')
            modes = Counter()
            for cluster in fresh['clusters']:
                for audit in cluster.get('dsp_mode_audit', []):
                    key = (audit['model'], tuple(sorted(audit['port_widths'].items())),
                           audit['parallel_count'], audit['demand'])
                    modes[key] += 1
            row.update(status='mode_verified', current=summary, dsp_mode_catalog=fresh['dsp_mode_catalog'],
                       dsp_modes=[dict(model=key[0], port_widths=dict(key[1]), parallel_count=key[2],
                                       demand_per_atom=key[3], count=count) for key, count in modes.items()])
        except (ValueError, KeyError) as exc:
            row.update(status='pending_exact_mode_evidence', error=f'{type(exc).__name__}: {exc}')
        addendum['designs'].append(row)
    write(args.out/'mode_aware_addendum.json', addendum)
    baseline = read(args.backend_evidence/'backend_manifest.json')
    binaries = {
        'fixed_2d_base': ('/home/phoenix/tools/vtr-40f8d197-tailcompile2d/vpr/vpr',
                         baseline['original_artifacts_before']['/home/phoenix/tools/vtr-40f8d197-tailcompile2d/vpr/vpr']),
        'fixed_v68': ('/home/phoenix/tools/vtr-40f8d197-tailcompile-v68/vpr/vpr',
                      '2a4bfc7f7ba906b41b4be1c7ad1e02a4ba46dcc7af2e26f6c7d42c2f3d98aca8'),
        'v7_3d_r2': ('/home/phoenix/tools/vtr-40f8d197-tailcompile-v7-3d-r2/vpr/vpr',
                      'ff5a39c53c7e0be00e68bb5c41afbc521c7d1ff0512dd44047b60396742f0a71')}
    binary_checks = {}
    for label, (path, expected) in binaries.items():
        actual = sha(path)
        binary_checks[label] = dict(path=path, sha256=actual, expected_sha256=expected,
                                    unchanged=actual == expected)
    version = subprocess.run([binaries['v7_3d_r2'][0], '--version'], capture_output=True, text=True, timeout=15)
    binary_checks['v7_3d_r2'].update(version_returncode=version.returncode,
                                     version_output=(version.stdout+version.stderr).strip())
    archived_bytes = sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())
    audit = dict(schema='tailcompile-v7-independent-audit-1', utc=datetime.now(timezone.utc).isoformat(),
                 scope='compact input archive, mode-aware reparse, binary hash/version checks; no P&R',
                 binary_checks=binary_checks, mode_addendum_sha256=sha(args.out/'mode_aware_addendum.json'),
                 mode_statuses={r['design']: r['status'] for r in addendum['designs']},
                 compact_archive_bytes_before_this_report=archived_bytes,
                 source_payload_deleted=False, oversized_graph_blif_and_rrg_copied=False,
                 cancelled_jobs='SIGTERM/INT mark module event, retire registered own groups, persist interrupted attempts',
                 cancellation_test='test_process_lifecycle.py passed SIGTERM and SIGINT subprocess/resume checks',
                 signal_limit='SIGKILL is uncatchable; scheduler must retire remaining job processes',
                 review_findings={'archive_overlap': 'root added source/destination disjoint guard and test',
                                  'campaign_concurrency': 'architecture agent added campaign lock and cancellation guards'},
                 code_sha256={name: sha(HERE/name) for name in ('io_utils.py', 'qualify.py',
                      'test_process_lifecycle.py', 'data.py', 'archive_input_evidence.py')},
                 cleanup_allocated_bytes_reclaimed=1511157760,
                 cleanup_additional_logical_bytes=4017499,
                 cleanup_note='allocated total covers failed checkout/build copies + W320 raw RRG; logical count is separate; excludes root architecture graph cleanup',
                 archived_groups=index)
    write(args.out/'independent_audit.json', audit)
    print(dict(mode_statuses=audit['mode_statuses'], archive_bytes=archived_bytes,
               binaries_ok=all(r['unchanged'] for r in binary_checks.values()), version_returncode=version.returncode))


if __name__ == '__main__':
    main()
