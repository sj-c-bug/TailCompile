"""Inventory the existing cohort and every top-level Koios RTL benchmark.

Discovery is performed against the pinned VTR checkout, not a guessed benchmark
count. Parameter variants are retained as designs but grouped into families.
"""
from __future__ import annotations

from pathlib import Path
import re

from bridge import V68, read, sha

SAFE_NAME = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\Z')
HELPERS = {'hard_block_include.v', 'hard_block_include.sv'}


def family_for(design, known):
    if design in known:
        return known[design]['family']
    # The frozen v68 family policy explicitly keeps these parameter variants
    # together. In particular, clstm/dla must not become independent samples.
    rules = ((r'^(?:clstm|dla)(?:[_.-]|$)', 'dla'),
             (r'^tpu(?:[_.-]|$)', 'tpu'), (r'^bwave(?:[_.-]|$)', 'bwave'),
             (r'^tdarknet(?:[_.-]|$)', 'tdarknet'),
             (r'^stereovision\d*(?:[_.-]|$)', 'stereovision'),
             (r'^diffeq\d*(?:[_.-]|$)', 'diffeq'),
             (r'^LU\d', 'LU'))
    for pattern, family in rules:
        if re.match(pattern, design, re.I):
            return family
    for name in sorted(known, key=lambda x: (-len(x), x)):
        if design.startswith(name + '.') or design.startswith(name + '_') or design.startswith(name + '-'):
            return known[name]['family']
    # Dot suffixes in Koios name size/dataflow variants. Unknown stems are
    # conservatively one family pending a source-lineage audit.
    return design.split('.', 1)[0]


def discover(vtr, source=None, checkpoint_metadata=None, designs=None, scope='all'):
    if scope not in {'all', 'koios', 'existing'}:
        raise ValueError(f'Unknown benchmark scope: {scope}')
    vtr = Path(vtr).resolve()
    bench = vtr / 'vtr_flow/benchmarks/verilog'
    koios = bench / 'koios'
    if not koios.is_dir():
        raise FileNotFoundError(f'Pinned VTR Koios benchmark directory missing: {koios}')
    base = read(V68 / 'cohort.json')['designs']
    old_cohort = Path(source) / 'cohort.json' if source else None
    if old_cohort is not None and old_cohort.is_file():
        extra = read(old_cohort).get('designs', [])
        by_name = {r['design']: dict(r) for r in base}
        for row in extra:
            if row['design'] in by_name and row['source'] != by_name[row['design']]['source']:
                raise ValueError(f'Conflicting source for existing benchmark: {row["design"]}')
            by_name[row['design']] = dict(row)
        base = list(by_name.values())
    known = {r['design']: r for r in base}
    candidates = {r['design']: dict(r, inventory_origin='existing_cohort') for r in base}
    excluded = []
    for path in sorted(koios.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {'.v', '.sv'}:
            continue
        if path.name in HELPERS:
            excluded.append(dict(source=path.relative_to(bench).as_posix(), reason='shared_hard_block_include', sha256=sha(path)))
            continue
        name = path.stem
        if not SAFE_NAME.fullmatch(name):
            raise ValueError(f'Unsafe discovered benchmark name: {name!r}')
        relative = path.relative_to(bench).as_posix()
        if name in candidates and candidates[name]['source'] != relative:
            raise ValueError(f'Discovered design name has two sources: {name}')
        if name in candidates:
            candidates[name]['inventory_origin'] = 'existing_cohort_and_koios'
        else:
            candidates[name] = dict(design=name, family=family_for(name, known),
                                    split='new', source=relative, inventory_origin='koios')
    requested = set(designs or [])
    if requested - candidates.keys():
        raise ValueError(f'Unknown requested designs: {sorted(requested-candidates.keys())}')
    meta = checkpoint_metadata or {}
    train = set(meta.get('training_families', []))
    validation = set(meta.get('validation_families', []))
    records = []
    for name, row in sorted(candidates.items()):
        in_scope = (scope == 'all' or
                    scope == 'koios' and row['source'].startswith('koios/') or
                    scope == 'existing' and name in known)
        if not in_scope:
            if name in requested:
                raise ValueError(f'Requested design {name} is outside scope={scope}')
            excluded.append(dict(design=name, source=row['source'], reason='outside_selected_scope'))
            continue
        if requested and name not in requested:
            excluded.append(dict(design=name, source=row['source'], reason='not_in_explicit_design_subset'))
            continue
        if not SAFE_NAME.fullmatch(name):
            raise ValueError(f'Unsafe existing benchmark name: {name!r}')
        path = (bench / row['source']).resolve()
        path.relative_to(bench.resolve())
        original_split = row.get('split', 'new')
        family = row['family']
        if family in train:
            exposure = 'checkpoint_training_family'
        elif family in validation:
            exposure = 'checkpoint_validation_family'
        else:
            old_splits = {r['split'] for r in base if r['family'] == family}
            exposure = ('previously_observed_holdout_family' if 'final_test' in old_splits else
                        'previous_training_family' if 'train' in old_splits else
                        'previous_validation_family' if 'validation' in old_splits else
                        'no_prior_exposure_recorded')
        include = path.parent / 'hard_block_include.v' if row['source'].startswith('koios/') else None
        missing = []
        if not path.is_file():
            missing.append(str(path))
        if include is not None and not include.is_file():
            missing.append(str(include))
        records.append(dict(row, split='sweep', original_split=original_split,
            checkpoint_exposure=exposure, source_path=str(path),
            source_sha256=sha(path) if path.is_file() else None,
            include_sha256=sha(include) if include is not None and include.is_file() else None,
            input_status='missing_input' if missing else 'available', missing_inputs=missing,
            lineage_status='existing_family_policy' if family in {r['family'] for r in base} else 'conservative_name_group_requires_lineage_audit'))
    return dict(schema='tailcompile-v610-benchmark-inventory-1', designs=records, excluded=excluded,
        benchmark_root=str(bench), design_count=len(records),
        family_count=len({r['family'] for r in records}),
        koios_design_count=sum(r['source'].startswith('koios/') for r in records),
        scope=scope,
        discovery_scope='Existing cohort union every top-level Koios .v/.sv, excluding shared hard-block includes',
        interpretation='Frozen-weight exploratory coverage sweep, including previously trained/observed families; not a new independent holdout',
        family_policy='Parameter/dataflow/size variants stay in one family; count designs and families separately')


discovery = discover
