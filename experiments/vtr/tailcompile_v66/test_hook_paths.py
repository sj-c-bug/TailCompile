import unittest
from pathlib import Path
from unittest.mock import MagicMock
from install_hook import verify_changes, REL


class HookPaths(unittest.TestCase):
    def root(self, target, symlink=True, exists=True):
        root=MagicMock()
        root.resolve.return_value=Path('/vtr')
        entry=root.__truediv__.return_value
        entry.is_symlink.return_value=symlink
        entry.is_file.return_value=exists
        entry.resolve.return_value=Path(target)
        return root

    def test_supported_build_outputs(self):
        for build in ('build','build_tailcompile_conda'):
            verify_changes(self.root(f'/vtr/{build}/vpr/vpr'), [REL,'vpr/vpr'])

    def test_source_edits_still_rejected(self):
        with self.assertRaises(ValueError):
            verify_changes(self.root('/vtr/build/vpr/vpr'),[REL,'vpr/src/other.cpp'])

    def test_external_dangling_or_plain_file_rejected(self):
        for root in (self.root('/elsewhere/vpr'), self.root('/vtr/build/vpr/vpr',False),
                     self.root('/vtr/build/vpr/vpr',exists=False)):
            with self.assertRaises(ValueError): verify_changes(root,[REL,'vpr/vpr'])

    def test_original_hook_only_supported(self):
        verify_changes(self.root('/vtr/build/vpr/vpr'),[REL])


if __name__=='__main__': unittest.main()
