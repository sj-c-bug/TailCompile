"""Numerical prior contract, scoped installation, and normal-build adapter."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import build_layer_backend as build
import install_layer_hook as hook
from layer_backend_audit import node_usage
from io_utils import sha


class LayerHookTests(unittest.TestCase):
    def test_observed_capacity_weighted_tail_and_overuse(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            p=Path(folder)/'usage.tsv'
            p.write_text('# iteration=1\n1\t1\t2\n2\t9\t0\n')
            result=node_usage(p)
            self.assertEqual(result['capacity'],10)
            self.assertEqual(result['overuse'],1)
            self.assertEqual(result['capacity_weighted_cvar95'],2.)
            p.write_text('1\t1\t2\n1\t1\t2\n')
            with self.assertRaises(ValueError): node_usage(p)

    def test_native_z_only_and_zero_release_transform(self):
        # The installed code hooks only the Z solve, and preserves native XY calls.
        text=Path(hook.__file__).read_text()
        self.assertIn('tailcompile_layer_apply(A_sparse_z, b_z',text)
        self.assertNotIn('tailcompile_layer_apply(A_sparse_x',text)
        self.assertNotIn('tailcompile_layer_apply(A_sparse_y',text)
        self.assertIn('tailcompile_layer_prior && i < 3',text)

    def test_normal_make_flags_and_link_read_only(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            base=Path(folder); b=base/'build'; work=b/'vpr'
            flags=work/'CMakeFiles/libvpr.dir'; flags.mkdir(parents=True)
            link=work/'CMakeFiles/vpr.dir'; link.mkdir(parents=True)
            cache=b/'CMakeCache.txt'; cache.write_text('CMAKE_CXX_COMPILER:FILEPATH=/compiler\n')
            (flags/'flags.make').write_text('CXX_DEFINES = -DEIGEN_INSTALLED\nCXX_INCLUDES = -I/source\nCXX_FLAGS = -O3 -std=c++20\n')
            (link/'link.txt').write_text('/compiler main.o -o vpr libvpr.a -lpthread\n')
            reference,command,cwd,archive=build.normal_commands(base,dict(build_directory=str(b),cmake_cache_sha256=sha(cache)))
            self.assertEqual(reference[0],'/compiler')
            self.assertIn('-DEIGEN_INSTALLED',reference)
            self.assertEqual(cwd,work)
            self.assertEqual(archive,work/'libvpr.a')
            self.assertEqual(command,['/compiler','main.o','-o','vpr','libvpr.a','-lpthread'])

    @unittest.skipUnless(shutil.which('c++') and Path('/usr/include/eigen3/Eigen/Sparse').exists(),
                         'requires the same C++/Eigen interface as fixed VTR')
    def test_exact_quadratic_addition_decay_fixed_exclusion_and_zero_identity(self):
        header=Path(hook.__file__).with_name('layer_prior_hook.inc').resolve()
        source=r'''
#include <cassert>
#include <stdexcept>
#include <cstdio>
#include <vector>
#include <Eigen/Sparse>
#define VPR_ERROR_AP 0
#define VPR_FATAL_ERROR(kind, ...) throw std::runtime_error("bad prior")
#define VTR_LOG(...) std::printf(__VA_ARGS__)
#include "HEADER"
struct Placement { std::vector<double> block_layer_nums; };
int main() {
  Placement p{{.2,.8,1.}};
  auto row_to_block=[](int i) { return i==0?0:2; }; // fixed block 1 has no solver row
  auto make=[]() { Eigen::SparseMatrix<double> a(2,2); a.insert(0,0)=2.; a.insert(1,1)=4.; return a; };
  setenv("TC_GP_INIT","fixture",1);
  setenv("TC_GP_LAYER_PRIOR","0",1);
  auto a=make(); Eigen::VectorXd b=Eigen::VectorXd::Zero(2);
  tailcompile_layer_apply(a,b,1,row_to_block);
  assert(a.coeff(0,0)==2. && a.coeff(1,1)==4. && b.squaredNorm()==0.);
  setenv("TC_GP_LAYER_PRIOR","1",1); tailcompile_layer_capture(p);
  tailcompile_layer_apply(a,b,1,row_to_block);
  assert(std::abs(a.coeff(0,0)-2.3)<1e-12 && std::abs(a.coeff(1,1)-4.3)<1e-12);
  assert(std::abs(b[0]-.06)<1e-12 && std::abs(b[1]-.3)<1e-12);
  a=make(); b.setZero(); tailcompile_layer_apply(a,b,2,row_to_block);
  assert(std::abs(a.coeff(0,0)-2.15)<1e-12 && std::abs(b[1]-.15)<1e-12);
  a=make(); b.setZero(); tailcompile_layer_apply(a,b,3,row_to_block);
  assert(a.coeff(0,0)==2. && a.coeff(1,1)==4. && b.squaredNorm()==0.);
  setenv("TC_GP_LAYER_PRIOR","2",1);
  try { tailcompile_layer_enabled(); return 9; } catch (const std::runtime_error&) {}
}
'''.replace('HEADER',str(header))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            src=Path(folder)/'prior.cpp'; exe=Path(folder)/'prior'
            src.write_text(source)
            subprocess.run(['c++','-std=c++20','-I/usr/include/eigen3',str(src),'-o',str(exe)],check=True,capture_output=True)
            result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
            self.assertIn('iteration=3 factor=0 beta=0',result.stdout)


if __name__=='__main__': unittest.main()
