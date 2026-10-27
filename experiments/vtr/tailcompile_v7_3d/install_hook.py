"""Install an exact 40f8 + heterogeneity-v2 + XYZ native-GP hook combination.

The pure v6.8 source transform is reused, never its installed binary or 2D hook.
Only pinned pristine files or the exact resulting combination are accepted.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'
FLOW = 'vpr/src/analytical_place/analytical_placement_flow.cpp'
GP = 'vpr/src/analytical_place/global_placement/global_placer.cpp'
SOLVER = 'vpr/src/analytical_place/global_placement/analytical_solver.cpp'
HOOK = 'vpr/src/analytical_place/warmstart_hook.inc'
HETERO_FILES = ('vpr/src/place/move_utils.cpp', 'vpr/src/place/initial_placement.cpp')
FILES = (FLOW, GP, SOLVER)
HETERO_SHA256 = 'e8a18c6924e7a8f87f8d6a41f532d166020da35cf475075662b0e4be1b9ed1d0'


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True)


def source_transform(pristine):
    source = Path(__file__).parents[1] / 'tailcompile_v68/install_hook.py'
    spec = importlib.util.spec_from_file_location('tailcompile_v68_exact_transform', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    transformed = {rel: value.replace('tailcompile_v68_', 'tailcompile_v7_')
                   .replace('TAILCOMPILE_V68_', 'TAILCOMPILE_V7_')
                   for rel, value in old.patch_sources(pristine).items()}
    before = '    full_legalizer->legalize(p_placement);\n'
    after = before + (
        '    VTR_LOG("TAILCOMPILE_V7_POST_APPACK grid=%zux%zux%zu\\n",\n'
        '            device_ctx.grid.width(), device_ctx.grid.height(), device_ctx.grid.get_num_layers());\n')
    transformed[FLOW] = old.once(transformed[FLOW], before, after)
    return transformed


def expected(root):
    if git(root, 'rev-parse', 'HEAD').strip() != COMMIT:
        raise ValueError(f'v7 requires fixed VTR {COMMIT}')
    return source_transform({rel: git(root, 'show', f'{COMMIT}:{rel}') for rel in FILES})


def heterogeneity_expected(root):
    """Apply each exact v2 hunk in memory, rejecting edits outside its scope."""
    root = Path(root)
    candidates = [root/'tailcompile_heterogeneity_v2.patch',
                  Path(__file__).parents[3]/'experiments/vtr3d/patches/vtr40f8_layer_heterogeneity_v2.patch']
    patch = next((path for path in candidates if path.is_file()), None)
    if patch is None:
        raise ValueError('provide audited tailcompile_heterogeneity_v2.patch in isolated root')
    if hashlib.sha256(patch.read_bytes()).hexdigest() != HETERO_SHA256:
        raise ValueError('heterogeneity v2 patch hash mismatch')
    text = patch.read_text()
    result = {}
    for section in text.split('diff --git ')[1:]:
        lines = section.splitlines(keepends=True)
        rel = lines[0].strip().split(' b/',1)[1]
        if rel not in HETERO_FILES:
            raise ValueError(f'unapproved heterogeneity patch path: {rel}')
        original = git(root,'show',f'{COMMIT}:{rel}').splitlines(keepends=True)
        cursor = 0
        output = []
        pos = 1
        while pos < len(lines):
            line = lines[pos]
            match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@',line)
            if not match:
                pos += 1
                continue
            start = int(match.group(1))-1
            if start < cursor:
                raise ValueError('overlapping heterogeneity hunks')
            output.extend(original[cursor:start])
            old, new = [], []
            pos += 1
            while pos < len(lines) and lines[pos][:1] in (' ','+','-'):
                item = lines[pos]
                if item[0] in (' ','-'):
                    old.append(item[1:])
                if item[0] in (' ','+'):
                    new.append(item[1:])
                pos += 1
            if original[start:start+len(old)] != old:
                raise ValueError(f'heterogeneity patch does not match pinned source: {rel}')
            output.extend(new)
            cursor = start+len(old)
        output.extend(original[cursor:])
        result[rel] = ''.join(output)
    if set(result) != set(HETERO_FILES):
        raise ValueError('heterogeneity patch must contain exactly both v2 files')
    return result


def verify(root):
    root = Path(root).resolve()
    wanted = {**expected(root), **heterogeneity_expected(root)}
    for rel, content in wanted.items():
        if (root/rel).read_text() != content:
            raise ValueError(f'v7 hook source mismatch: {rel}')
    hook = Path(__file__).with_name('warmstart_hook.inc').read_text()
    if (root/HOOK).read_text() != hook:
        raise ValueError('v7 XYZ warmstart_hook.inc mismatch')
    changed = git(root, 'diff', '--name-only', '--ignore-submodules', 'HEAD').splitlines()
    if set(changed) - set(FILES) - set(HETERO_FILES) - {'vpr/vpr'}:
        raise ValueError(f'unapproved changes in isolated v7 checkout: {changed}')
    payload = ''.join(rel+'\n'+wanted[rel] for rel in FILES)+hook
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    root = a.root.resolve()
    wanted = {**expected(root), **heterogeneity_expected(root)}
    if not a.verify:
        changed = git(root, 'diff', '--name-only', '--ignore-submodules', 'HEAD').splitlines()
        if set(changed) - set(FILES) - set(HETERO_FILES) - {'vpr/vpr'}:
            raise ValueError(f'unapproved source changes: {changed}')
        for rel, content in wanted.items():
            pristine = git(root, 'show', f'{COMMIT}:{rel}')
            if (root/rel).read_text() not in (pristine, content):
                raise ValueError(f'refusing unrelated edits in {rel}')
        hook = root/HOOK
        source = Path(__file__).with_name('warmstart_hook.inc').read_text()
        if hook.exists() and hook.read_text() != source:
            raise ValueError('refusing unknown existing hook')
        for rel, content in wanted.items():
            (root/rel).write_text(content)
        hook.write_text(source)
    print(json.dumps(dict(verified=True, vtr_commit=COMMIT, hook_sha256=verify(root),
                          coordinate_schema='tailcompile_v7_ap_graph',
                          raw_layer_to_discrete='std_round_nonnegative')))


if __name__ == '__main__':
    main()
