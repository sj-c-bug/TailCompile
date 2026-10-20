"""Fast installer and compiled validation-contract tests (no VPR labels needed)."""
import json
import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import sys
import time
import unittest
from pathlib import Path

import install_hook as hook


def real_vpr_smoke(root, out, baseline_vpr=None):
    """Optional tiny actual-VPR test, separate from synthetic contract tests.

    Example: python3 test_hook.py --real-vtr "$VTR_ROOT" --smoke-out /tmp/v68_smoke
    A fresh output directory is required; no experiment evidence is overwritten.
    The teacher replay is an interface check, NOT a learned-generator benchmark.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    hook_hash = hook.verify(root)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Real smoke output must be empty/new; existing evidence is never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    binary = root / 'vpr/vpr'
    arch = root / 'vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml'
    blif = root / 'vtr_flow/benchmarks/blif/4/5xp1.blif'
    base = [str(binary), str(arch), str(blif), '--analytical_place', '--route', '--analysis',
            '--ap_full_legalizer', 'appack', '--ap_detailed_placer', 'annealer',
            '--route_chan_width', '100', '--device_width', '14', '--num_workers', '1', '--seed', '1']
    observations = {}

    def run(name, env_extra=None, command=None, markers=()):
        folder = out / name
        folder.mkdir()
        env = {k: v for k, v in os.environ.items() if not k.startswith('TC_GP_')}
        env.update({k: str(v) for k, v in (env_extra or {}).items()})
        command = command or base
        started = time.perf_counter()
        result = subprocess.run(command, cwd=folder, env=env, capture_output=True,
                                text=True, timeout=180)
        text = result.stdout + result.stderr
        (folder / 'vpr.out').write_text(text)
        observations[name] = dict(returncode=result.returncode,
            seconds=time.perf_counter()-started, command=command,
            markers=[line for line in text.splitlines() if 'TAILCOMPILE' in line])
        if result.returncode or any(marker not in text for marker in markers):
            raise RuntimeError(f'Real VPR smoke failed: {folder}/vpr.out')
        return text

    prefix = out / 'structure/ap'
    structure_text = run('structure', dict(TC_GP_STRUCTURE=prefix, TC_GP_STRUCTURE_ONLY='1'),
        markers=['TAILCOMPILE_V68_STRUCTURE_ONLY_COMPLETE no_gp_executed=1'])
    if 'AP Global Placer' in structure_text:
        raise AssertionError('Structure extraction executed GP')
    structure_rows = [r.split('\t') for r in Path(str(prefix)+'.blocks.tsv').read_text().splitlines()]
    if not structure_rows or any(r[4:6] != ['-1', '-1'] for r in structure_rows if r[3]=='0'):
        raise AssertionError('Movable positions leaked into structure snapshot')
    run('default', dict(TC_GP_EXPORT=out/'default/teacher'),
        markers=['TAILCOMPILE_V68_EXPORTED stage=post_gp_teacher', 'VPR succeeded'])
    teacher = [r.split('\t') for r in (out/'default/teacher.blocks.tsv').read_text().splitlines()]
    prior = out/'init.ap.tsv'
    prior.write_text(''.join(' '.join([r[1],*r[4:8]])+'\n' for r in teacher if r[3]=='0'))
    text = run('warm', dict(TC_GP_INIT=prior), markers=[
        'TAILCOMPILE_V68_GP_INIT_CONSUMED',
        'TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1', 'VPR succeeded'])
    if 'skipping the Global Placement' in text:
        raise AssertionError('Warm-start bypassed native GP')
    command = [str(binary), str(arch), str(blif), '--pack', '--write_rr_graph',
               str(out/'pack_rrg/rr_graph.xml'), '--route_chan_width', '100',
               '--place_chan_width', '100', '--device_width', '14', '--num_workers', '1']
    run('pack_rrg', command=command, markers=['VPR succeeded'])
    rr_graph = out/'pack_rrg/rr_graph.xml'
    if not rr_graph.is_file():
        raise AssertionError('Pack-only run did not export RRG')
    identity = None
    if baseline_vpr:
        old_command = [str(Path(baseline_vpr).resolve()), *base[1:]]
        run('old_default', command=old_command, markers=['VPR succeeded'])
        def placements(path):
            return [line for line in path.read_text().splitlines()
                    if len(line.split())>=5 and line.split()[1].lstrip('-').isdigit()]
        before, after = placements(out/'old_default/5xp1.place'), placements(out/'default/5xp1.place')
        identity = before == after
        if not identity:
            raise AssertionError('Inactive patched VPR changed default placement')
    report = dict(status='passed', scope='Interface correctness only, not learned-model quality/speedup',
        vtr_commit=hook.COMMIT, hook_sha256=hook_hash,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        ap_blocks=len({r[0] for r in structure_rows}), structure_prefix=str(prefix),
        teacher_prefix=str(out/'default/teacher'), rr_graph=str(rr_graph),
        pack_only_rr_graph_bytes=rr_graph.stat().st_size,
        inactive_placement_identical_to_baseline=identity, trials=observations)
    (out/'smoke_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',report=str(out/'smoke_report.json'))))
    return report


HARNESS = r'''
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
struct Id { size_t n; operator size_t() const { return n; } bool is_valid() const {return true;} };
using AtomBlockId=Id; using APBlockId=Id; using PackMoleculeId=Id; using APNetId=Id; using APPinId=Id;
enum class APBlockMobility {MOVEABLE, FIXED};
enum class PinType {DRIVER, SINK};
struct Fixed { double x=1.5,y=1.5; int layer_num=0,sub_tile=0; };
struct AtomNetlist {
 std::string block_name(Id a) const {return std::vector<std::string>{"a","b","c","fixed"}[a];}
 Id block_model(Id) const {return {0};}
};
struct Mol {std::vector<Id> atom_block_ids;};
struct Prepacker {Mol get_molecule(Id m) const {return m.n==0?Mol{{{0},{1}}}:Mol{{{m.n+1}}};}};
struct APNetlist {
 std::vector<Id> blocks() const {return {{0},{1},{2}};}
 std::vector<Id> block_molecules(Id b) const {return {b};}
 APBlockMobility block_mobility(Id b) const {return b.n==2?APBlockMobility::FIXED:APBlockMobility::MOVEABLE;}
 Fixed block_loc(Id) const {return {};}
 std::vector<Id> nets() const {return {{0}};}
 bool net_is_ignored(Id) const {return false;}
 std::vector<Id> net_pins(Id) const {return {{0},{1},{2}};}
 Id pin_block(Id p) const {return p;}
 PinType pin_type(Id p) const {return p.n==0?PinType::DRIVER:PinType::SINK;}
};
struct PartialPlacement {
 std::vector<double> block_x_locs{-1.,-1.,1.5},block_y_locs{-1.,-1.,1.5},block_layer_nums{0.,0.,0.};
 std::vector<int> block_sub_tiles{0,0,0};
 explicit PartialPlacement(const APNetlist&) {}
};
struct Grid {size_t width() const{return 4;} size_t height() const{return 4;} size_t get_num_layers() const{return 1;}};
struct Models {std::string model_name(Id) const{return ".names";}};
struct Arch {Models models;};
struct Device {Grid grid; Arch object; const Arch* arch=&object;};
struct Flat {bool valid=false;};
struct AtomCtx {Flat flat_placement_info() const{return {};}};
struct Ctx {Device d; const Device& device()const{return d;} AtomCtx atom()const{return {};}} g_vpr_ctx;
#define VPR_ERROR_AP 0
#define VPR_FATAL_ERROR(kind,...) throw std::runtime_error("validation rejected")
#define VTR_LOG(...) std::printf(__VA_ARGS__)
#include "warmstart_hook.inc"
int main(int argc,char** argv) {
 try {
  APNetlist ap; AtomNetlist atoms; Prepacker pp; PartialPlacement p(ap);
  if(std::string(argv[1])=="structure") {tailcompile_v68_structure(atoms,ap,pp); return 3;}
  bool used=tailcompile_v68_load_initial(atoms,ap,pp,p);
  if(std::string(argv[1])=="inactive") return used?4:0;
  if(!used || p.block_x_locs[0]!=1.5 || p.block_x_locs[1]!=2.5 || p.block_x_locs[2]!=1.5) return 5;
  tailcompile_v68_export(argv[2],"post_gp_teacher",atoms,ap,pp,p);
  return 0;
 } catch (const std::exception& error) {std::fprintf(stderr,"%s\n",error.what());return 2;}
}
'''


class HookSourceTest(unittest.TestCase):
    def fixture(self):
        return {
            hook.FLOW: '/**\n * @brief A helper\n*/\n'
                       '    if (g_vpr_ctx.atom().flat_placement_info().valid) {\n}\n'
                       '    print_ap_netlist_stats(ap_netlist);\n'
                       '    // Verify that the partial placement is valid before running the full\n',
            hook.GP: '#include "vtr_time.h"\n'
                     '    PartialPlacement p_placement(ap_netlist_);\n'
                     '    for (size_t i = 0; i < max_num_iterations_; i++) {\n'
                     '        solver_->solve(i, p_placement);\n'
                     '        // Exit condition: If the upper-bound and lower-bound HPWLs are\n}\n',
            hook.SOLVER: '#include <cstdio>\n'
                         'void B2BSolver::solve(unsigned iteration, PartialPlacement& p_placement) {\n'
                         '    // untouched native solve body\n}\n',
        }

    def test_patch_keeps_native_loop_and_guarantees_followup_solve(self):
        result = hook.patch_sources(self.fixture())
        self.assertIn('solver_->solve(i, p_placement);', result[hook.GP])
        self.assertIn('if (tailcompile_warm_started && i == 0) continue;', result[hook.GP])
        self.assertIn('TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED', result[hook.GP])
        self.assertIn('iteration == 0 && std::getenv("TC_GP_INIT")', result[hook.SOLVER])
        self.assertIn('// untouched native solve body', result[hook.SOLVER])
        self.assertIn('tailcompile_v68_structure', result[hook.FLOW])
        self.assertIn('LP_B2B', result[hook.FLOW])

    def test_layout_mismatch_is_rejected(self):
        fixture = self.fixture()
        fixture[hook.GP] += '    PartialPlacement p_placement(ap_netlist_);\n'
        with self.assertRaises(ValueError):
            hook.patch_sources(fixture)

    def test_unrelated_modified_source_is_rejected(self):
        with self.assertRaises(ValueError):
            hook.verify_changes(Path('.'), ['vpr/src/route/route.cpp'])


@unittest.skipUnless(shutil.which('c++') or shutil.which('g++'), 'C++ compiler unavailable')
class CompiledHookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='tailcompile_v68_hook_')
        cls.root = Path(cls.temp.name)
        for name in ('logic_types.h', 'prepack.h'):
            (cls.root / name).write_text('// mock type declarations are in harness\n')
        shutil.copyfile(Path(__file__).with_name('warmstart_hook.inc'), cls.root / 'warmstart_hook.inc')
        (cls.root / 'harness.cpp').write_text(HARNESS)
        cls.binary = cls.root / 'harness'
        subprocess.run([shutil.which('c++') or shutil.which('g++'), '-std=c++17', '-Wall', '-Wextra',
                        str(cls.root / 'harness.cpp'), '-I', str(cls.root), '-o', str(cls.binary)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def invoke(self, text=None, mode='load', **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith('TC_GP_')}
        if text is not None:
            path = self.root / 'prior.fplace'
            path.write_text(text)
            env['TC_GP_INIT'] = str(path)
        env.update(extra)
        return subprocess.run([str(self.binary), mode, str(self.root / 'teacher')],
                              env=env, capture_output=True, text=True)

    def test_complete_coherent_prior_and_fixed_omission(self):
        r = self.invoke('a 1.5 1.5 0 0\nb 1.5 1.5 0 0\nc 2.5 2.5 0 0\n')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('GP_INIT_CONSUMED', r.stdout)
        schema = json.loads((self.root / 'teacher.schema.json').read_text())
        self.assertEqual(schema['stage'], 'post_gp_teacher')
        self.assertEqual(len((self.root / 'teacher.blocks.tsv').read_text().splitlines()[0].split('\t')), 10)

    def test_missing_split_unknown_outside_and_fixed_motion_rejected(self):
        base = 'a 1.5 1.5 0 0\nb 1.5 1.5 0 0\nc 2.5 2.5 0 0\n'
        bad = [base.replace('b 1.5 1.5 0 0\n', ''),
               base.replace('b 1.5', 'b 1.7'), base + 'unknown 1 1 0 0\n',
               base.replace('c 2.5', 'c 4'), base + 'fixed 2.5 1.5 0 0\n',
               base + 'a 1.5 1.5 0 0\n']
        for text in bad:
            with self.subTest(text=text):
                self.assertEqual(self.invoke(text).returncode, 2)

    def test_inactive_hook_does_not_change_native_initialization(self):
        self.assertEqual(self.invoke(mode='inactive').returncode, 0)

    def test_structure_only_stops_before_gp_and_exports_no_moving_locations(self):
        prefix = str(self.root / 'structure')
        r = self.invoke(mode='structure', TC_GP_STRUCTURE=prefix, TC_GP_STRUCTURE_ONLY='1')
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('STRUCTURE_ONLY_COMPLETE no_gp_executed=1', r.stdout)
        rows = [line.split('\t') for line in Path(prefix + '.blocks.tsv').read_text().splitlines()]
        self.assertTrue(all(row[4:6] == ['-1', '-1'] for row in rows if row[3] == '0'))


if __name__ == '__main__':
    if '--real-vtr' in sys.argv:
        parser = argparse.ArgumentParser(description='Real tiny VPR native-GP interface smoke')
        parser.add_argument('--real-vtr', type=Path, required=True)
        parser.add_argument('--smoke-out', type=Path, required=True)
        parser.add_argument('--baseline-vpr', type=Path)
        args = parser.parse_args()
        real_vpr_smoke(args.real_vtr,args.smoke_out,args.baseline_vpr)
    else:
        unittest.main()
