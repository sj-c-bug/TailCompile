"""Stream atom-matched layer observations without keeping stage atom maps.

There is one initial atom->index dictionary. Later stages use compact arrays
and a byte mask, so exporting a large atom placement does not create another
full string-keyed placement dictionary per stage. A missing export is missing
evidence, never zero movement.
"""
from __future__ import annotations

from array import array
from collections import Counter
import hashlib
import math
from pathlib import Path

SCHEMA = 'tailcompile-v7-layer-observations-1'


def _lines(path, digest):
    with Path(path).open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            yield raw.decode('utf-8').strip()


def _layer(value, *, continuous):
    z=float(value)
    if not math.isfinite(z) or z<0:
        raise ValueError('layer observation needs finite nonnegative Z')
    # VTR's declared std_round_nonnegative mapping differs from round(.5).
    discrete=math.floor(z+.5)
    if not continuous and not math.isclose(z,discrete,abs_tol=1e-6):
        raise ValueError('flat placement requires an integer layer')
    return discrete


def _flat_rows(path, digest):
    for line in _lines(path,digest):
        if not line or line.startswith('#'):
            continue
        row=line.split()
        if len(row)<5 or (len(row)>5 and not row[5].startswith('#')):
            raise ValueError(f'expected atom x y layer subtile in {path}')
        # Native verbosity 1/2 appends '#clbid atom_pb_type [site_path]'.
        # Only fields after the five coordinates are comments: an atom can
        # itself contain '#' and must retain its complete identity.
        name,x,y,z,sub=row[:5]
        if not all(math.isfinite(float(v)) and float(v)>=0 for v in (x,y)):
            raise ValueError('flat placement observation has invalid XY')
        if not float(sub).is_integer() or int(float(sub))<0:
            raise ValueError('flat placement observation has invalid subtile')
        yield name,_layer(z,continuous=False),None


def _gp_rows(path, digest):
    for line in _lines(path,digest):
        if not line or line.startswith('#'):
            continue
        row=line.split('\t')
        if len(row)!=10:
            raise ValueError(f'expected ten native AP block columns in {path}')
        raw,name,model,fixed,x,y,z,sub,model_name,molecule=row
        if fixed not in ('0','1'):
            raise ValueError('native AP fixed flag must be 0/1')
        yield name,_layer(z,continuous=True),bool(int(fixed))


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def collect_layers(init_path, post_gp_prefix, final_fplace=None, legalized_fplace=None):
    """Observe init->GP->APPack legalization->final atom layers.

    ``post_gp_prefix.blocks.tsv`` is the native ten-column AP snapshot. Flat
    exports have name/x/y/layer/subtile columns, optionally followed by native
    '#clbid atom_pb_type [site_path]' diagnostic fields. The legalized export is
    after APPack+full legalization, before detailed placement; it is not a
    separate observation of raw packing alone. Fixed coverage derives solely
    from native AP mobility flags, not an inferred fixed-Z constraint.
    """
    index={}
    initial_layers=array('i')
    initial_hash=None
    initial_counts=Counter()
    init_path=Path(init_path) if init_path is not None else None
    if init_path is not None and init_path.is_file():
        digest=hashlib.sha256()
        for name,z,fixed in _flat_rows(init_path,digest):
            if name in index:
                raise ValueError('duplicate initial atom in layer observations')
            index[name]=len(index)
            initial_layers.append(z)
            initial_counts[str(z)]+=1
        initial_hash=digest.hexdigest()
    n=len(index)
    fixed_flags=bytearray(n)  # 0 unknown, 1 native movable, 2 native fixed.
    previous=array('i',initial_layers)
    previous_stage='initial'
    observations={}
    gp_path=Path(str(post_gp_prefix)+'.blocks.tsv') if post_gp_prefix is not None else None
    paths=(('post_gp',gp_path,True),
           ('post_appack_legalization',Path(legalized_fplace) if legalized_fplace is not None else None,False),
           ('final',Path(final_fplace) if final_fplace is not None else None,False))
    for stage,path,is_gp in paths:
        if path is None or not path.is_file():
            observations[stage]=dict(status='missing',reason='not_requested' if path is None else 'file_not_found',
                source=str(path.resolve()) if path is not None else None,source_sha256=None,
                matched_initial_atoms=None,matched_fraction_of_initial=None,changed_fraction=None,
                layer_counts=None,fixed_flag_coverage=None)
            continue
        digest=hashlib.sha256()
        current=array('i',[-1])*n
        seen=bytearray(n)
        counts=Counter();transitions=Counter();previous_transitions=Counter()
        matched=changed=rows=unknown=previous_matched=previous_changed=0
        fixed_known=fixed_matched=fixed_changed=0
        native_fixed_rows=unmatched_native_fixed=unmatched_native_movable=0
        unknown_sample=[]
        for atom,z,flag in (_gp_rows(path,digest) if is_gp else _flat_rows(path,digest)):
            rows+=1
            counts[str(z)]+=1
            native_fixed_rows+=flag is True
            i=index.get(atom)
            if i is None:
                unknown+=1
                unmatched_native_fixed+=flag is True
                unmatched_native_movable+=flag is False
                if len(unknown_sample)<16: unknown_sample.append(atom)
                continue
            if seen[i]:
                raise ValueError(f'duplicate matched atom in {stage} observation: {atom}')
            seen[i]=1;matched+=1;current[i]=z
            moved=z!=initial_layers[i]
            changed+=moved
            transitions[f'{initial_layers[i]}->{z}']+=1
            if previous[i]>=0:
                previous_matched+=1
                previous_changed+=z!=previous[i]
                previous_transitions[f'{previous[i]}->{z}']+=1
            if flag is not None:
                fixed_flags[i]=2 if flag else 1
            if fixed_flags[i]:
                fixed_known+=1
            if fixed_flags[i]==2:
                fixed_matched+=1;fixed_changed+=moved
        observations[stage]=dict(status='observed' if matched==n and n>0 and unknown==0 else 'partial_coverage',
            source=str(path.resolve()),source_sha256=digest.hexdigest(),atom_rows=rows,
            matched_initial_atoms=matched,missing_initial_atoms=n-matched,unmatched_atom_rows=unknown,
            unmatched_atom_sample=unknown_sample,
            native_fixed_rows=native_fixed_rows if is_gp else None,
            unmatched_native_fixed_rows=unmatched_native_fixed if is_gp else None,
            unmatched_native_movable_rows=unmatched_native_movable if is_gp else None,
            matched_fraction_of_initial=matched/n if n else None,
            changed_atoms=changed,changed_fraction=changed/matched if matched else None,
            changed_fraction_denominator='matched initial atoms',layer_counts=dict(sorted(counts.items())),
            matched_transitions_from_initial=dict(sorted(transitions.items())),
            previous_available_stage=previous_stage,matched_previous_atoms=previous_matched,
            changed_fraction_from_previous=previous_changed/previous_matched if previous_matched else None,
            matched_transitions_from_previous=dict(sorted(previous_transitions.items())),
            fixed_flag_known_matched_atoms=fixed_known,
            fixed_flag_coverage=fixed_known/matched if matched else None,
            native_fixed_matched_atoms=fixed_matched,native_fixed_layer_changes=fixed_changed,
            native_fixed_changed_fraction=fixed_changed/fixed_matched if fixed_matched else None,
            missing_fixed_is_not_zero=True)
        previous=current
        previous_stage=stage
    fixed_counts=Counter(str(initial_layers[i]) for i,flag in enumerate(fixed_flags) if flag==2)
    return dict(schema=SCHEMA,implementation_sha256=_hash(__file__),
        logical_unit='atom',initial=dict(status='observed' if n else 'missing_or_empty',
            source=str(init_path.resolve()) if init_path is not None else None,source_sha256=initial_hash,
            atom_count=n,layer_counts=dict(sorted(initial_counts.items())),
            fixed_flag_known_atoms=sum(bool(f) for f in fixed_flags),
            fixed_flag_coverage=sum(bool(f) for f in fixed_flags)/n if n else None,
            native_fixed_atom_count=sum(f==2 for f in fixed_flags),native_fixed_layer_counts=dict(sorted(fixed_counts.items()))),
        stages=observations,
        gp_layer_mapping='floor(z+0.5), equivalent to std::round for nonnegative native Z',
        fixed_semantics='native AP mobility FIXED flag; fixed-Z constraints not independently supplied',
        appack_boundary='APPack plus full legalization, before detailed placement, when export is supplied',
        limitations=['unmatched atoms are reported, not assumed stationary',
                     'atom-weighted changes are not independent APBlock samples',
                     'native fixed-flag changes are observations, not proof of fixed-Z constraint violations'])
