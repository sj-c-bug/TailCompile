"""Compiled XYZ input-contract tests, independent of model training or route QoR."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import install_hook


class XYZContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++') or shutil.which('g++')
        if not compiler:
            raise unittest.SkipTest('requires native C++ compiler; execute this test in WSL/Linux')
        cls.tmp = tempfile.TemporaryDirectory(prefix='tailcompile-v7-hook-')
        cls.root = Path(cls.tmp.name)
        old = Path(__file__).parents[1]/'tailcompile_v68/test_hook.py'
        parsed = ast.parse(old.read_text())
        harness = next(ast.literal_eval(node.value) for node in parsed.body
                       if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HARNESS' for t in node.targets))
        harness = (harness.replace('tailcompile_v68_', 'tailcompile_v7_')
                   .replace('get_num_layers() const{return 1;}', 'get_num_layers() const{return 2;}')
                   .replace('int layer_num=0,sub_tile=0;', 'int layer_num=1,sub_tile=0;')
                   .replace('block_layer_nums{0.,0.,0.}', 'block_layer_nums{0.,0.,1.}'))
        (cls.root/'main.cpp').write_text(harness)
        for name in ('logic_types.h','prepack.h'):
            (cls.root/name).write_text('')
        shutil.copy2(Path(__file__).with_name('warmstart_hook.inc'),cls.root/'warmstart_hook.inc')
        cls.binary=cls.root/'hook_test'
        subprocess.run([compiler,'-std=c++17','-O0','-I',str(cls.root),str(cls.root/'main.cpp'),'-o',str(cls.binary)],check=True,capture_output=True,text=True)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'):
            cls.tmp.cleanup()

    def run_case(self, rows, name='input'):
        source=self.root/(name+'.tsv')
        source.write_text(rows)
        env={key:value for key,value in os.environ.items() if not key.startswith('TC_GP_')}
        env['TC_GP_INIT']=str(source)
        return subprocess.run([str(self.binary),'warm',str(self.root/name)],env=env,capture_output=True,text=True)

    @staticmethod
    def valid(z=1):
        return f'a 1.5 1.5 {z} 0\nb 1.5 1.5 {z} 0\nc 2.5 2.5 0 0\n'

    def test_nonzero_layer_accepted_and_schema_exported(self):
        r=self.run_case(self.valid(),'nonzero')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('nonzero_layer_blocks=1',r.stdout)
        schema=json.loads((self.root/'nonzero.schema.json').read_text())
        self.assertEqual(schema['layer_count'],2)
        self.assertEqual(schema['raw_layer_to_discrete'],'std_round_nonnegative')
        self.assertEqual((self.root/'nonzero.meta.tsv').read_text().split(),['4','4','2'])

    def test_checksum_changes_when_only_layer_changes(self):
        import re
        a=self.run_case(self.valid(0),'layer0')
        b=self.run_case(self.valid(1),'layer1')
        self.assertEqual(a.returncode,0,a.stderr)
        self.assertEqual(b.returncode,0,b.stderr)
        self.assertNotEqual(re.search(r'checksum=([^ ]+)',a.stdout).group(1),re.search(r'checksum=([^ ]+)',b.stdout).group(1))

    def test_fractional_initial_layer_rejected(self):
        self.assertEqual(self.run_case(self.valid(0.5),'fraction').returncode,2)

    def test_invalid_layer_rejected(self):
        for z in (-1,2,'nan'):
            self.assertEqual(self.run_case(self.valid(z),'invalid_'+str(z)).returncode,2)

    def test_molecule_split_layer_rejected(self):
        self.assertEqual(self.run_case(self.valid().replace('b 1.5 1.5 1','b 1.5 1.5 0'),'split').returncode,2)

    def test_unknown_or_missing_atom_rejected(self):
        for rows in (self.valid()+'unknown 1 1 0 0\n',self.valid().replace('b 1.5 1.5 1 0\n','')):
            self.assertEqual(self.run_case(rows,'coverage').returncode,2)

    def test_fixed_xyz_cannot_be_changed(self):
        self.assertEqual(self.run_case(self.valid()+'fixed 1.5 1.5 0 0\n','fixed_bad').returncode,2)
        self.assertEqual(self.run_case(self.valid()+'fixed 1.5 1.5 1 0\n','fixed_ok').returncode,0)

    def test_duplicate_atom_rejected(self):
        self.assertEqual(self.run_case(self.valid()+'a 1.5 1.5 1 0\n','duplicate').returncode,2)


if __name__=='__main__':
    unittest.main()
