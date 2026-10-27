"""Keep 3D writes out of frozen 2D modules and execution dependencies."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def _within(path, root):
    return path == root or root in path.parents


def validate_output_roots(paths, *, repo=None, dependency_roots=()):
    """Resolve symlinks and reject overlapping source/dependency write roots.

    Local standalone scratch is allowed. Inside the canonical repository,
    learning results belong under this version's runs/ and scratch under
    vtr_work/tailcompile_v7_3d/. Other versions are read-only dependencies.
    """
    repo = Path(repo or REPO).resolve()
    versions = repo/'experiments/vtr'
    own = versions/HERE.name
    protected = {versions/'tailcompile_v68', versions/'tailcompile_v69'}
    if versions.is_dir():
        protected.update(p for p in versions.iterdir() if p.is_dir() and p.name != HERE.name)
    protected.update(Path(p).resolve() for p in dependency_roots if p is not None)
    protected = {p.resolve() for p in protected}
    for candidate in paths:
        path = Path(candidate).resolve()
        if any(_within(path, root) or _within(root, path) for root in protected):
            raise ValueError(f'3D output overlaps a protected 2D/VTR root: {path}')
        if _within(path, own) and not _within(path, own/'runs'):
            raise ValueError(f'3D results cannot overwrite source files; use this version runs/: {path}')
        scratch = repo/'vtr_work'
        if _within(path, scratch) and not _within(path, scratch/HERE.name):
            raise ValueError(f'3D scratch requires its own version namespace: {path}')
    return [Path(p).resolve() for p in paths]
