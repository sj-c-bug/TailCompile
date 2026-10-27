"""Bounded original-flow ABC mapping and actual pre-GP hard-resource census.

Two registered primary train families and one backup only. No GP, placement,
route, GNN inference or architecture primitive extension is performed.
"""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys

from architectures import COMMIT,SOURCE_ARCH,_module,generate_architecture
from data import build_logic
from io_utils import read,write,sha,run_trial,vpr_command


PRIMARY=(('mcml','mcml',64),('ch_intrinsics','ch_intrinsics',64))
BACKUP=('stereovision3','stereovision',64)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('vtr-root','synthesis-root','binary','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for name in ('vtr_root','synthesis_root','binary','out'):
        setattr(a,name,getattr(a,name).resolve())
    a.out.mkdir(parents=True,exist_ok=True)
    source_commit=subprocess.check_output(['git','-C',str(a.vtr_root),'rev-parse','HEAD'],text=True).strip()
    synth_commit=subprocess.check_output(['git','-C',str(a.synthesis_root),'rev-parse','HEAD'],text=True).strip()
    if source_commit!=COMMIT:
        raise ValueError('input RTL source must be the fixed 40f8 commit')
    synth_tools=[a.synthesis_root/'build/bin/yosys',a.synthesis_root/'build/share/yosys/plugins/parmys.so',
                 a.synthesis_root/'build/abc/abc']
    tools_before={str(path):sha(path) for path in synth_tools}
    report=dict(schema='tailcompile-v7-heavy-input-probe-1',source_vtr_commit=source_commit,
                synthesis_tool_commit=synth_commit,synthesis_tools_sha256=tools_before,
                structure_binary_sha256=sha(a.binary),minimum_hard_units=16,
                primary=[x[0] for x in PRIMARY],backup=BACKUP[0],candidates=[],
                scope='ABC-only mapping followed by TC_GP_STRUCTURE_ONLY; no GP/P&R/GNN')
    def candidate(spec):
        design,family,grid=spec
        folder=a.out/design
        folder.mkdir(exist_ok=True)
        source=a.vtr_root/'vtr_flow/benchmarks/verilog'/(design+'.v')
        arch=folder/'arch.xml'
        if not arch.exists():
            write(folder/'architecture.json',generate_architecture(a.vtr_root/SOURCE_ARCH,arch,0,0,grid,grid))
        metadata=read(folder/'architecture.json')
        row=dict(design=design,family=family,split='train',grid=[grid,grid,2],
                 source=str(source),source_sha256=sha(source),architecture_sha256=sha(arch),
                 full_gp_or_route_executed=False)
        try:
            command=[sys.executable,str(a.synthesis_root/'vtr_flow/scripts/run_vtr_flow.py'),str(source),str(arch),
                     '-temp_dir',str(folder/'mapping'),'-ending_stage','abc','-timeout','120']
            mapped=run_trial(folder/'synthesis',command,inputs=[source,arch,*synth_tools],timeout=180,prune=False,
                metadata=dict(stage='parmys_abc_only',source_commit=source_commit,synthesis_commit=synth_commit))
            row['synthesis_status']=mapped['status']
            row['synthesis_log']=mapped['log']
            if mapped['status']!='complete':
                text=Path(mapped['log']).read_text(errors='replace')
                row['wrapper_status']=mapped['status']
                if 'returncode  :  -15' in text and 'Executable yosys failed' in text:
                    row['synthesis_status']='timeout'
                    row['synthesis_timeout_seconds']=120
                    row['timeout_evidence']='original flow configured -timeout 120; Yosys terminated with returncode -15'
                raise RuntimeError('original-flow mapping did not complete; hard-resource demand remains unknown')
            raw=folder/'mapping'/(design+'.abc.blif')
            if not raw.is_file():
                raise FileNotFoundError('ABC output missing')
            blif=folder/'design.blif'
            if not blif.exists():
                write(folder/'mapping_adaptation.json',_module('prepare_natural_blif').prepare_blif(raw,blif,arch))
            prefix=folder/'structure'
            structure=run_trial(folder/'structure_run',vpr_command(a.binary,arch,blif,32,701),
                inputs=[a.binary,arch,blif],env=dict(TC_GP_STRUCTURE=prefix,TC_GP_STRUCTURE_ONLY=1),
                timeout=180,marker='TAILCOMPILE_V7_STRUCTURE_ONLY_COMPLETE',
                metadata=dict(stage='pre_gp_resource_census'))
            row['structure_status']=structure['status']
            row['structure_log']=structure['log']
            if structure['status']!='complete':
                raise RuntimeError('actual APBlock structure export failed')
            text=Path(structure['log']).read_text(errors='replace')
            if '## AP Global Placer' in text or 'Circuit successfully routed' in text:
                raise AssertionError('structure-only scope unexpectedly ran GP or route')
            logic=build_logic(prefix,design,blif,arch,family=family,split='train')
            write(folder/'logic.json',logic)
            demand=[sum(c['demand'][axis] for c in logic['clusters']) for axis in range(5)]
            available=metadata['capacity_total']
            row.update(structure_compatible=True,ap_blocks=len(logic['clusters']),demand_keys=logic['demand_keys'],
                demand=demand,hard_units=demand[2]+demand[3],
                heavy_count_threshold_passed=demand[2]+demand[3]>=16,
                ram_fraction_of_grid=demand[2]/available['memory'],
                dsp_fraction_of_grid=demand[3]/available['dsp_top'],
                physical_capacity=available,derived_blif=str(blif),derived_blif_sha256=sha(blif),
                ap_primitive_capacity_is_proxy=True)
        except Exception as exc:
            row.update(structure_compatible=None if row.get('synthesis_status')!='complete' else False,
                       error=f'{type(exc).__name__}: {exc}')
        report['candidates'].append(row)
        write(a.out/'heavy_input_audit.json',report)
        print(row,flush=True)
    for spec in PRIMARY:
        candidate(spec)
    if not any(row.get('heavy_count_threshold_passed') for row in report['candidates']):
        candidate(BACKUP)
    report['synthesis_tools_unchanged']=tools_before=={str(path):sha(path) for path in synth_tools}
    report['complete']=True
    report['usable_heavy_candidates']=[row['design'] for row in report['candidates']
        if row.get('structure_compatible') and row.get('heavy_count_threshold_passed')]
    report['remaining_qualification']='Grid utilization must be judged against predeclared resource criteria; Default P&R and graph qualification remain pending.'
    write(a.out/'heavy_input_audit.json',report)
    print(dict(complete=True,usable=report['usable_heavy_candidates'],tools_unchanged=report['synthesis_tools_unchanged']),flush=True)


if __name__=='__main__':
    main()
