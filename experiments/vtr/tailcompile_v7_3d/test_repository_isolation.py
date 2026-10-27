"""Read-only frozen 2D helpers and isolated 3D runtime/namespace contracts.

Use fresh processes, as the CLI does. Discovery across multiple version folders
in one interpreter is deliberately not a supported execution contract: those
legacy CLIs expose the same bare names (data/model/train/io_utils).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = Path(__file__).resolve().parent


class RepositoryIsolationTests(unittest.TestCase):
    def test_output_guard_allows_only_versioned_repo_results_and_scratch(self):
        from repository_guard import validate_output_roots
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root/'repo'
            versions = repo/'experiments/vtr'
            for name in ('tailcompile_v68', 'tailcompile_v69', 'tailcompile_v7_3d', 'another_version'):
                (versions/name).mkdir(parents=True)
            allowed = [versions/'tailcompile_v7_3d/runs/cohort',
                       repo/'vtr_work/tailcompile_v7_3d/cohort', root/'external_scratch']
            self.assertEqual(validate_output_roots(allowed, repo=repo), [p.resolve() for p in allowed])
            forbidden = [versions/'tailcompile_v68/runs', versions/'tailcompile_v69/model',
                         versions/'another_version/output', versions/'tailcompile_v7_3d/new_source',
                         versions, repo, root, repo/'vtr_work', repo/'vtr_work/other_version']
            for path in forbidden:
                with self.subTest(path=path), self.assertRaises(ValueError):
                    validate_output_roots([path], repo=repo)

    def test_output_guard_rejects_vtr_mapping_roots_and_symlink_aliases(self):
        from repository_guard import validate_output_roots
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root/'repo'
            protected = [root/'vtr_execution', root/'vtr_mapping']
            for path in protected:
                path.mkdir()
                with self.subTest(path=path), self.assertRaises(ValueError):
                    validate_output_roots([path/'runs'], repo=repo, dependency_roots=protected)
                with self.subTest(ancestor=path.parent), self.assertRaises(ValueError):
                    validate_output_roots([path.parent], repo=repo, dependency_roots=protected)
            alias = root/'innocent_output_alias'
            try:
                alias.symlink_to(protected[0], target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest('host does not permit directory symlinks')
            with self.assertRaises(ValueError):
                validate_output_roots([alias/'results'], repo=repo, dependency_roots=protected)

    def run_probe(self, body):
        isolated = HERE/'runs'
        isolated.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='isolation_', dir=isolated) as temporary:
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                               MKL_THREADING_LAYER='SEQUENTIAL')
            result = subprocess.run([sys.executable, '-B', '-c', textwrap.dedent(body),
                                     str(HERE), temporary], cwd=HERE, env=environment,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
            return json.loads(result.stdout.strip().splitlines()[-1])

    def test_qualified_dependencies_ignore_foreign_bare_model_and_data(self):
        row = self.run_probe('''
            import importlib, json, pathlib, sys, types
            import torch
            here = pathlib.Path(sys.argv[1]); sys.path.insert(0, str(here.parent))
            poison = types.ModuleType('foreign_2d_module')
            poison.__file__ = '/foreign_2d_sentinel.py'
            def forbidden(name):
                raise AssertionError('foreign bare model/data used: '+name)
            poison.__getattr__ = forbidden
            sys.modules['model'] = poison; sys.modules['data'] = poison
            model = importlib.import_module('tailcompile_v7_3d.model')
            train = importlib.import_module('tailcompile_v7_3d.train')
            inherited = importlib.import_module('tailcompile_v68.model')
            assert model.Hierarchical3DInitializer.__mro__[1] is inherited.HierarchicalInitializer
            assert train.Hierarchical3DInitializer is model.Hierarchical3DInitializer
            assert sys.modules['model'] is poison and sys.modules['data'] is poison
            print(json.dumps(dict(qualified_dependencies=True, foreign_bare_modules_untouched=True)))
        ''')
        self.assertTrue(row['qualified_dependencies'])

    def test_3d_training_and_lowering_do_not_write_frozen_2d_trees(self):
        row = self.run_probe('''
            import hashlib, importlib, json, os, pathlib, sys
            import torch
            torch.set_num_threads(1)
            here, output = map(pathlib.Path, sys.argv[1:])
            sys.path.insert(0, str(here.parent))
            protected = [here.parent/'tailcompile_v68', here.parent/'tailcompile_v69']
            sentinel = output/'frozen_sentinel'; sentinel.mkdir(); protected.append(sentinel)
            def hashes():
                return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                        for root in protected for p in root.glob('*.py')}
            before = hashes()
            def check(path):
                if not isinstance(path, (str, bytes, os.PathLike)):
                    return
                resolved = pathlib.Path(os.fsdecode(path)).resolve()
                if any(resolved == root or root in resolved.parents for root in protected):
                    raise PermissionError('frozen dependency mutation: '+str(resolved))
            def audit(event, args):
                if event == 'open':
                    path, mode, flags = args
                    if ((mode and any(c in mode for c in 'wax+')) or
                            flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):
                        check(path)
                elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime','os.truncate'):
                    check(args[0])
                elif event == 'os.rename':
                    check(args[0]); check(args[1])
            sys.addaudithook(audit)
            try:
                (sentinel/'blocked.py').write_text('must not be written')
                raise AssertionError('audit protection did not block write')
            except PermissionError:
                assert not (sentinel/'blocked.py').exists()
            model = importlib.import_module('tailcompile_v7_3d.model')
            train = importlib.import_module('tailcompile_v7_3d.train')
            data = importlib.import_module('tailcompile_v7_3d.data')
            io = importlib.import_module('tailcompile_v7_3d.io_utils')
            lower = importlib.import_module('tailcompile_v7_3d.generation').lower_initialization
            assert data.atom_resource.__module__ == 'tailcompile_v68.data'
            assert io.parse_phase_metrics.__module__ == 'tailcompile_v69.phase_metrics'
            assert pathlib.Path(data.read.__code__.co_filename).resolve() == here/'io_utils.py'
            demand = torch.tensor([[1.,0.,0.,0.,0.]]*3)
            graph = dict(cluster_x=torch.ones(3,3), region_x=torch.ones(4,5),
                logic_src=torch.tensor([0,1]), logic_dst=torch.tensor([1,2]),
                logic_weight=torch.ones(2), region_src=torch.tensor([0,1,2,3]),
                region_dst=torch.tensor([1,0,3,2]), region_weight=torch.ones(4),
                region_bounds=torch.tensor([[0.,.5,0.,1.],[.5,1.,0.,1.]]*2),
                region_layer=torch.tensor([0,0,1,1]), demand=demand,
                capacity=torch.tensor([[2.,2.,1.,2.,2.]]*4),
                pool_x=torch.ones(2,8), pool_ids=torch.tensor([0,0,1,1]),
                pool_regions=torch.tensor([0,2,1,3]), parents=[torch.tensor([0,0,1,1])],
                movable=torch.ones(3,dtype=torch.bool), fixed_region=torch.full((3,),-1),
                fixed_layer=torch.full((3,),-1), fixed_xy=torch.full((3,2),-1.),
                channel_width=64., teacher_axis_scale=[16.,16.])
            prediction = model.capacity_only(graph)
            teacher = dict(teacher_region=prediction['assignment'].tolist(), teacher_xy=prediction['xy'].tolist())
            report = train.train([dict(graph=graph,teacher=teacher,split='train',family='isolation',
                                       condition_id='isolation')], output/'training', epochs=1,hidden=8)
            logical = dict(clusters=[dict(atoms=['atom'+str(i)]) for i in range(3)])
            generated = lower(logical,dict(grid_width=16,grid_height=16,layers=2), prediction,
                              output/'initialization',method='capacity_only')
            assert pathlib.Path(generated['ap_init']).is_relative_to(output)
            assert all(pathlib.Path(path).is_relative_to(output) for path in report['checkpoints'])
            assert before == hashes()
            print(json.dumps(dict(frozen_2d_unchanged=True, weights_changed=report['weights_changed'],
                                  outputs_under_3d_temporary_root=True)))
        ''')
        self.assertTrue(row['frozen_2d_unchanged'])
        self.assertTrue(row['outputs_under_3d_temporary_root'])


if __name__ == '__main__':
    unittest.main()
