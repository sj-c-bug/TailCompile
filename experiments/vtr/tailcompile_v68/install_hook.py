"""Install the audited v6.8 native-GP warm initialization hook (not GP bypass)."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'
FLOW = 'vpr/src/analytical_place/analytical_placement_flow.cpp'
GP = 'vpr/src/analytical_place/global_placement/global_placer.cpp'
SOLVER = 'vpr/src/analytical_place/global_placement/analytical_solver.cpp'
HOOK = 'vpr/src/analytical_place/warmstart_hook.inc'
FILES = (FLOW, GP, SOLVER)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode()


def once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f'unexpected pinned source layout: {before[:100]!r}')
    return source.replace(before, after, 1)


def patch_sources(pristine):
    """Pure transformation; only exact commit sources are accepted by installer."""
    flow = once(pristine[FLOW], '/**\n * @brief A helper',
                '#include "warmstart_hook.inc"\n\n/**\n * @brief A helper')
    flow = once(flow, '    print_ap_netlist_stats(ap_netlist);\n',
                '    print_ap_netlist_stats(ap_netlist);\n'
                '    tailcompile_v68_structure(atom_nlist, ap_netlist, prepacker);\n')
    flow = once(flow, '    if (g_vpr_ctx.atom().flat_placement_info().valid) {\n',
                '    if (std::getenv("TC_GP_INIT")\n'
                '        && ap_opts.analytical_solver_type != e_ap_analytical_solver::LP_B2B)\n'
                '        VPR_FATAL_ERROR(VPR_ERROR_AP, "TC_GP_INIT requires the native LP_B2B solver");\n'
                '    if (g_vpr_ctx.atom().flat_placement_info().valid) {\n')
    flow = once(flow, '    // Verify that the partial placement is valid before running the full\n',
                '    tailcompile_v68_export(std::getenv("TC_GP_EXPORT"), "post_gp_teacher",\n'
                '                          atom_nlist, ap_netlist, prepacker, p_placement);\n\n'
                '    // Verify that the partial placement is valid before running the full\n')
    gp = once(pristine[GP], '#include "vtr_time.h"\n',
              '#include "vtr_time.h"\n#include "../warmstart_hook.inc"\n')
    gp = once(gp, '    PartialPlacement p_placement(ap_netlist_);\n',
              '    PartialPlacement p_placement(ap_netlist_);\n'
              '    const bool tailcompile_warm_started = tailcompile_v68_load_initial(\n'
              '        atom_netlist_, ap_netlist_, prepacker_, p_placement);\n')
    gp = once(gp, '        solver_->solve(i, p_placement);\n',
              '        solver_->solve(i, p_placement);\n'
              '        if (tailcompile_warm_started && i == 1)\n'
              '            VTR_LOG("TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1\\n");\n')
    gp = once(gp, '        // Exit condition: If the upper-bound and lower-bound HPWLs are\n',
              '        // The learned seed is not an optimized lower bound. Never use its\n'
              '        // first legalization gap to skip the actual native B2B iterations.\n'
              '        if (tailcompile_warm_started && i == 0) continue;\n\n'
              '        // Exit condition: If the upper-bound and lower-bound HPWLs are\n')
    solver = once(pristine[SOLVER], '#include <cstdio>\n',
                  '#include <cstdio>\n#include <cstdlib>\n')
    solver = once(solver, 'void B2BSolver::solve(unsigned iteration, PartialPlacement& p_placement) {\n',
                  'void B2BSolver::solve(unsigned iteration, PartialPlacement& p_placement) {\n'
                  '    // A validated learned prior replaces ONLY the cold initialization.\n'
                  '    // Iteration zero still executes native legalization/timing; iterations\n'
                  '    // one onward execute the unmodified B2B solver and native anchors.\n'
                  '    if (iteration == 0 && std::getenv("TC_GP_INIT")) {\n'
                  '        block_x_locs_solved = p_placement.block_x_locs;\n'
                  '        block_y_locs_solved = p_placement.block_y_locs;\n'
                  '        if (is_multi_die()) block_z_locs_solved = p_placement.block_layer_nums;\n'
                  '        VTR_LOG("TAILCOMPILE_V68_B2B_WARM_START native_gp_continues=1\\n");\n'
                  '        return;\n'
                  '    }\n')
    return {FLOW: flow, GP: gp, SOLVER: solver}


def pristine_sources(root):
    if git(root, 'rev-parse', 'HEAD').strip() != COMMIT:
        raise ValueError(f'VTR commit mismatch: v6.8 requires {COMMIT}')
    return {rel: git(root, 'show', f'{COMMIT}:{rel}') for rel in FILES}


def verify_changes(root, changed):
    if set(changed) - set(FILES) - {'vpr/vpr'}:
        raise ValueError(f'unexpected VTR source modifications: {changed}')
    if 'vpr/vpr' in changed:
        entry = root / 'vpr/vpr'
        allowed = {root.resolve() / build / 'vpr/vpr'
                   for build in ('build', 'build_tailcompile_conda', 'build_tailcompile_v68')}
        if not entry.is_symlink() or not entry.is_file() or entry.resolve() not in allowed:
            raise ValueError('vpr/vpr must link to an existing VPR binary in an audited build directory')


def verify(root):
    root = Path(root).resolve()
    expected = patch_sources(pristine_sources(root))
    for rel, wanted in expected.items():
        if (root / rel).read_text() != wanted:
            raise ValueError(f'install the exact v6.8 warm-start hook and rebuild VPR: {rel}')
    content = Path(__file__).with_name('warmstart_hook.inc').read_text()
    if not (root / HOOK).is_file() or (root / HOOK).read_text() != content:
        raise ValueError('v6.8 warmstart_hook.inc mismatch')
    verify_changes(root, git(root, 'diff', '--name-only', '--ignore-submodules', 'HEAD').splitlines())
    payload = ''.join(rel + '\n' + expected[rel] for rel in FILES) + content
    return hashlib.sha256(payload.encode()).hexdigest()


def legacy_expected(root):
    """Recognize exact old GP export hook; do not silently accept arbitrary edits."""
    old = Path(__file__).parents[1] / 'tailcompile_v66/install_hook.py'
    spec = importlib.util.spec_from_file_location('tailcompile_v66_audited_hook', old)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.expected(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.verify:
        print(json.dumps({'verified': True, 'hook_sha256': verify(root), 'vtr_commit': COMMIT}))
        return
    pristine = pristine_sources(root)
    desired = patch_sources(pristine)
    verify_changes(root, git(root, 'diff', '--name-only', '--ignore-submodules', 'HEAD').splitlines())
    for rel in FILES:
        allowed = {pristine[rel], desired[rel]}
        if rel == FLOW:
            allowed.add(legacy_expected(root))
        current = (root / rel).read_text()
        if current not in allowed:
            raise ValueError(f'unrelated source edits; refusing to overwrite {rel}')
    hook = root / HOOK
    content = Path(__file__).with_name('warmstart_hook.inc').read_text()
    if hook.exists() and hook.read_text() != content:
        raise ValueError('unknown existing warmstart_hook.inc; refusing to overwrite')
    # Existing exact v6.6 flow can be upgraded; original source remains in Git.
    for rel, source in desired.items():
        (root / rel).write_text(source)
    hook.write_text(content)
    print(json.dumps({'installed': True, 'hook_sha256': verify(root),
                      'vtr_commit': COMMIT, 'rebuild_required': True,
                      'native_gp_warmstart': True}))


if __name__ == '__main__':
    main()
