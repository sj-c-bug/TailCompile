"""Exact isolated 40f8 + v7 + decaying Z prior + observational routing hook.

The existing installer and all historical v7 entry points remain untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import install_hook as old

COMMIT = old.COMMIT
GP, SOLVER, FLOW = old.GP, old.SOLVER, old.FLOW
ROUTE = 'vpr/src/route/route.cpp'
PRIOR = 'vpr/src/analytical_place/global_placement/layer_prior_hook.inc'
ROUTE_HOOK = 'vpr/src/route/layer_route_hook.inc'
FILES = (*old.FILES, *old.HETERO_FILES, ROUTE)


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f'pinned layer hook anchor not unique: {before[:90]!r}')
    return text.replace(before, after, 1)


def source_transform(pristine):
    result = old.source_transform(pristine)
    gp = result[GP]
    gp = once(gp, '    float total_time_spent_in_solver = 0.0f;\n',
              '    const bool tailcompile_layer_prior = tailcompile_layer_enabled()\n'
              '        && g_vpr_ctx.device().grid.get_num_layers() > 1;\n'
              '    float total_time_spent_in_solver = 0.0f;\n')
    gp = once(gp, '#include "../warmstart_hook.inc"\n',
              '#include "../warmstart_hook.inc"\n#include "layer_prior_hook.inc"\n')
    gp = once(gp, '        if (tailcompile_warm_started && i == 0) continue;\n',
              '        if (tailcompile_warm_started && i == 0) continue;\n'
              '        // Complete a zero-prior native solve before considering normal exit.\n'
              '        if (tailcompile_layer_prior && i < 3) continue;\n')
    gp = once(gp, '    // Update the setup slacks. This is performed down here (as well as being\n',
              '    if (tailcompile_layer_prior)\n'
              '        VTR_LOG("TAILCOMPILE_V7_LAYER_PRIOR_RELEASED native_exit=1\\n");\n'
              '    // Update the setup slacks. This is performed down here (as well as being\n')
    solver = result[SOLVER]
    solver = once(solver, '#include "vtr_vector.h"\n',
                  '#include "vtr_vector.h"\n#include "layer_prior_hook.inc"\n')
    solver = once(solver, '    if (iteration == 0 && std::getenv("TC_GP_INIT")) {\n',
                  '    if (iteration == 0 && std::getenv("TC_GP_INIT")) {\n'
                  '        tailcompile_layer_capture(p_placement);\n')
    solver = once(solver, '            z = solve_linear_system(A_sparse_z, b_z, z_guess);\n',
                  '            tailcompile_layer_apply(A_sparse_z, b_z, iteration,\n'
                  '                [&](int row) { return row_id_to_blk_id_[APRowId(row)]; });\n'
                  '            z = solve_linear_system(A_sparse_z, b_z, z_guess);\n')
    route = once(pristine[ROUTE], '#include "vtr_expr_eval.h"\n',
                 '#include "vtr_expr_eval.h"\n#include "layer_route_hook.inc"\n')
    route = once(route, '        bool routing_is_feasible = feasible_routing();\n',
                 '        tailcompile_layer_route_usage(itry);\n'
                 '        bool routing_is_feasible = feasible_routing();\n')
    result.update({GP: gp, SOLVER: solver, ROUTE: route})
    return result


def expected(root):
    root = Path(root)
    overlay = root/'source_overlay.json'
    base = Path(json.loads(overlay.read_text())['base_root']) if overlay.exists() else root
    if old.git(base, 'rev-parse', 'HEAD').strip() != COMMIT:
        raise ValueError(f'layer backend requires {COMMIT}')
    result = source_transform({p: old.git(base, 'show', f'{COMMIT}:{p}')
                               for p in (*old.FILES, ROUTE)})
    result.update(old.heterogeneity_expected(base))
    here = Path(__file__).parent
    result[old.HOOK] = (here/'warmstart_hook.inc').read_text()
    result[PRIOR] = (here/'layer_prior_hook.inc').read_text()
    result[ROUTE_HOOK] = (here/'layer_route_hook.inc').read_text()
    return result


def verify(root):
    root = Path(root).resolve()
    wanted = expected(root)
    for rel, text in wanted.items():
        if (root/rel).read_text() != text:
            raise ValueError(f'layer backend source mismatch: {rel}')
    if (root/'source_overlay.json').exists():
        overlay = json.loads((root/'source_overlay.json').read_text())
        if overlay.get('vtr_commit') != COMMIT or old.verify(Path(overlay['base_root'])) != overlay['base_hook_sha256']:
            raise ValueError('read-only overlay base changed')
        actual = {str(p.relative_to(root)) for p in (root/'vpr/src').rglob('*') if p.is_file()}
        if actual != set(wanted):
            raise ValueError('unapproved source overlay files')
    else:
        changed = old.git(root, 'diff', '--name-only', '--ignore-submodules', 'HEAD').splitlines()
        if set(changed) - set(FILES) - {'vpr/vpr'}:
            raise ValueError(f'unapproved layer backend source changes: {changed}')
    return hashlib.sha256(''.join(p+'\n'+wanted[p] for p in sorted(wanted)).encode()).hexdigest()


def install(root):
    root = Path(root).resolve()
    from repository_guard import validate_output_roots
    validate_output_roots((root,))
    wanted = expected(root)
    previous = old.expected(root)
    previous.update(old.heterogeneity_expected(root))
    for rel, text in wanted.items():
        path = root/rel
        if path.exists():
            allowed = {text}
            if rel in FILES:
                allowed.add(old.git(root, 'show', f'{COMMIT}:{rel}'))
                if rel in previous: allowed.add(previous[rel])
            elif rel == old.HOOK:
                allowed.add(Path(__file__).with_name('warmstart_hook.inc').read_text())
            if path.read_text() not in allowed:
                raise ValueError(f'refusing unknown edits: {rel}')
    for rel, text in wanted.items():
        (root/rel).parent.mkdir(parents=True, exist_ok=True)
        (root/rel).write_text(text)
    return verify(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    value = verify(a.root) if a.verify else install(a.root)
    print(json.dumps(dict(verified=True, hook_sha256=value, vtr_commit=COMMIT,
                          schema='tailcompile-layer-backend-007', rebuild_required=not a.verify)))


if __name__ == '__main__': main()
