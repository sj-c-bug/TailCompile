"""Install a narrowly scoped, reversible hook into the pinned server VTR tree."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'
REL = 'vpr/src/analytical_place/analytical_placement_flow.cpp'
MARK = '    // Verify that the partial placement is valid before running the full\n'
INCLUDE = '#include "tailcompile_gp_hook.inc"\n'
CALL = '    tailcompile_gp_hook(atom_nlist, ap_netlist, prepacker, p_placement);\n\n'

def verify_changes(root, changed):
    """Accept audited source plus a build symlink; binary hashes remain frozen."""
    if set(changed) - {REL, 'vpr/vpr'}:
        raise ValueError(f'unexpected VTR source modifications: {changed}')
    if 'vpr/vpr' in changed:
        entry = root / 'vpr/vpr'
        allowed = {root.resolve() / build / 'vpr/vpr'
                   for build in ('build', 'build_tailcompile_conda')}
        if not entry.is_symlink() or not entry.is_file() or entry.resolve() not in allowed:
            raise ValueError('vpr/vpr must link to an existing build/vpr/vpr or '
                             'build_tailcompile_conda/vpr/vpr inside this VTR tree')


def expected(root):
    pristine = subprocess.check_output(['git', '-C', str(root), 'show', f'{COMMIT}:{REL}']).decode()
    if pristine.count(MARK) != 1:
        raise ValueError('unexpected pinned source layout')
    return pristine.replace('/**\n * @brief A helper', INCLUDE + '\n/**\n * @brief A helper', 1).replace(MARK, CALL + MARK)

def verify(root):
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip() != COMMIT:
        raise ValueError('VTR commit mismatch')
    target = root / REL
    hook = target.with_name('tailcompile_gp_hook.inc')
    wanted = Path(__file__).with_name('gp_hook.inc').read_text()
    if target.read_text() != expected(root) or not hook.is_file() or hook.read_text() != wanted:
        raise ValueError('install the exact v6.6 GP hook and rebuild VPR first')
    changed = subprocess.check_output(['git', '-C', str(root), 'diff', '--name-only', '--ignore-submodules', 'HEAD']).decode().splitlines()
    verify_changes(root, changed)
    return hashlib.sha256(wanted.encode()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--upgrade', action='store_true', help='preserve prior hook before replacing it')
    a = p.parse_args()
    root = a.root.resolve()
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD']).decode().strip()
    if actual != COMMIT:
        raise ValueError(f'expected {COMMIT}, got {actual}')
    changed=subprocess.check_output(['git','-C',str(root),'diff','--name-only','--ignore-submodules','HEAD']).decode().splitlines()
    verify_changes(root, changed)
    target = root / REL
    pristine = subprocess.check_output(['git', '-C', str(root), 'show', f'{COMMIT}:{REL}']).decode()
    desired = expected(root)
    if target.read_text() not in (pristine, desired):
        raise ValueError('source already contains unrelated edits; refusing to overwrite')
    hook = target.with_name('tailcompile_gp_hook.inc')
    content = Path(__file__).with_name('gp_hook.inc').read_text()
    if hook.exists() and hook.read_text() != content:
        if not a.upgrade:
            raise ValueError('existing hook differs; use --upgrade to preserve and update it')
        old = hook.read_bytes()
        hook.with_suffix('.backup-' + hashlib.sha256(old).hexdigest()[:12]).write_bytes(old)
    hook.write_text(content)
    target.write_text(desired)
    print(json.dumps({'installed': True, 'hook_sha256': verify(root), 'rebuild_required': True}))

if __name__ == '__main__':
    main()
