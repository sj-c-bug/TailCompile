"""Write the frozen manifest for an isolated normal CMake backend build."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import install_hook


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1024*1024),b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--build',type=Path,required=True)
    a=p.parse_args()
    root,build=a.root.resolve(),a.build.resolve()
    if root not in build.parents:
        p.error('CMake build must be inside isolated source root')
    binary=root/'vpr/vpr'
    cache=build/'CMakeCache.txt'
    if not binary.is_file() or not cache.is_file():
        p.error('completed VPR binary and CMake cache required')
    source={rel:sha(root/rel) for rel in (*install_hook.FILES,*install_hook.HETERO_FILES,install_hook.HOOK)}
    manifest=dict(schema='tailcompile-v7-backend-1',vtr_commit=install_hook.COMMIT,
        build_method='normal_cmake_build',binary=str(binary.resolve()),binary_sha256=sha(binary),
        source_sha256=source,hook_sha256=install_hook.verify(root),
        heterogeneity_patch_sha256=sha(root/'tailcompile_heterogeneity_v2.patch'),
        cmake_cache_sha256=sha(cache),build_directory=str(build),
        build_commands=[['cmake','--build',str(build),'--target','vpr']],
        coordinate_schema='tailcompile_v7_ap_graph',raw_layer_to_discrete='std_round_nonnegative')
    target=root/'backend_manifest.json'
    if target.exists():
        old=json.loads(target.read_text())
        if old!=manifest:
            raise ValueError('backend manifest already freezes a different build; use a new isolated root')
    else:
        temporary=target.with_name(target.name+f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(manifest,indent=2)+'\n')
        temporary.replace(target)
    print(json.dumps(dict(backend_manifest=str(target),binary_sha256=manifest['binary_sha256'])))


if __name__=='__main__':
    main()
