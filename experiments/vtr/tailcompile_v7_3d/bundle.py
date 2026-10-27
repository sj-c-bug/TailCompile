"""Historical offline archive export; normal synchronization uses TailCompile Git."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import zipfile
from architectures import LEGACY
from io_utils import sha,write

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def bundle(output,example_sha=None):
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite a released bundle: {output}')
    entries={}
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py','.sh','.md','.json','.inc'):
            entries[str(path.relative_to(REPO)).replace('\\','/')]=path
    for path in HERE.parent.glob('*.py'):
        entries[str(path.relative_to(REPO)).replace('\\','/')]=path
    legacy_v68=HERE.with_name('tailcompile_v68')
    for path in legacy_v68.iterdir():
        if path.is_file() and path.suffix in ('.py','.sh','.json','.md','.inc'):
            entries[str(path.relative_to(REPO)).replace('\\','/')]=path
    phase=HERE.with_name('tailcompile_v69')/'phase_metrics.py'
    entries[str(phase.relative_to(REPO)).replace('\\','/')]=phase
    for name in ('heterogeneity_fixtures.py','pilot_architectures.py','device_supply_3d.py',
                 'prepare_natural_blif.py','probe_layer_heterogeneity.py','test_device_supply_3d.py'):
        entries['experiments/vtr3d/'+name]=LEGACY/name
    patch=LEGACY/'patches/vtr40f8_layer_heterogeneity_v2.patch'
    entries['experiments/vtr3d/patches/'+patch.name]=patch
    for name in ('EXPERIMENT_PLAN_3D_V68_2026-10-23.md','PRE_GNN_EXECUTION_PLAN_2026-10-23.md'):
        entries['experiments/vtr3d/'+name]=LEGACY/name
    for name in ('RESULTS_V7_PREPARATION_2026-10-25.md','RESULTS_ARCHITECTURES_2026-10-24.md'):
        if (LEGACY/name).is_file():
            entries['experiments/vtr3d/'+name]=LEGACY/name
    research = LEGACY.parents[1]/'research idea.md'
    if research.is_file():
        entries['research idea.md']=research
    if example_sha:
        entries['qualification_inputs/sha.pre-vpr.blif']=Path(example_sha)
    for name,path in entries.items():
        if not path.is_file():
            raise FileNotFoundError(f'missing bundle dependency {name}: {path}')
    # The source workspace nests implementation under TailCompile_latest/;
    # a server bundle is itself the repo root. Only these documentation links
    # change. Keep both source and packaged hashes for the audit.
    payloads={}
    for name,path in entries.items():
        payload=path.read_bytes()
        if path.suffix=='.md':
            value=payload.decode('utf-8')
            if name.startswith('experiments/vtr3d/'):
                value=value.replace('../../TailCompile_latest/experiments/vtr/','../vtr/')
            elif name=='research idea.md':
                value=value.replace('TailCompile_latest/experiments/vtr/','experiments/vtr/')
            payload=value.encode('utf-8')
        payloads[name]=payload
    import hashlib
    manifest=dict(schema='tailcompile-v7-server-source-bundle-2',
        files=[dict(path=name,source_sha256=sha(path),
                    sha256=hashlib.sha256(payloads[name]).hexdigest(),bytes=len(payloads[name]))
               for name,path in sorted(entries.items())],
        includes_vtr_binary=False,includes_large_datasets=False,includes_trained_checkpoint=False,
        documentation_link_rebase='TailCompile_latest/ prefix removed only in copied documentation links',
        historical_run_logs_included=False,
        extraction='Extract into one new repository directory; paths start with experiments/.',
        purpose='server CPU/GPU staged experiment; local engineering smoke is not a performance-accepted model')
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,payload in sorted(payloads.items()):
            archive.writestr(name,payload)
        archive.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    result=dict(bundle=str(output),sha256=sha(output),bytes=output.stat().st_size,
                source_file_count=len(entries),manifest=manifest)
    write(output.with_suffix('.manifest.json'),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--example-sha',type=Path)
    parser.add_argument('--legacy-export',action='store_true',
                        help='explicit historical/offline export only; not the server deployment workflow')
    args=parser.parse_args()
    if not args.legacy_export:
        parser.error('Use the main TailCompile Git repository and prepare_server.sh; see REPOSITORY_WORKFLOW.md. Offline export requires --legacy-export.')
    result=bundle(args.out,args.example_sha)
    print({key:value for key,value in result.items() if key!='manifest'})
