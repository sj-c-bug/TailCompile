"""Measure pre-GP demands; this probe does not run global placement or routing."""
import argparse
from pathlib import Path
from architectures import SOURCE_ARCH,generate_architecture,_module
from data import build_logic
from io_utils import freeze,write,run_trial,vpr_command,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','vtr-root','binary','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--designs',nargs='+',default=['sha','or1200','attention_layer','spmv'])
    args=parser.parse_args()
    if args.out.is_dir() and any(args.out.iterdir()) and not (args.out/'probe_protocol.json').is_file():
        raise ValueError('historical input probe has no current source freeze; use a new output directory')
    args.out.mkdir(parents=True,exist_ok=True)
    freeze(args.out/'probe_protocol.json',dict(schema='tailcompile-v7-input-probe-2',
        implementation_sha256=sha(__file__),data_sha256=sha(Path(__file__).with_name('data.py')),
        binary_sha256=sha(args.binary),designs=args.designs,source_root=str(args.source_root.resolve())))
    rows=[]
    cases=[('sha','sha','light',16),('or1200','or1200','light',20),
           ('attention_layer','attention','heavy',96),('spmv','spmv','heavy',64),
           ('tpu_like.small.os','tpu','heavy',64),('stereovision0','stereovision','heavy',32),
           ('bgm','bgm','heavy',64)]
    cases=[row for row in cases if row[0] in args.designs]
    if len(cases)!=len(args.designs):
        raise ValueError('unknown or duplicate design in registered input probe')
    for design,family,group,grid in cases:
        folder=args.out/design
        folder.mkdir(parents=True,exist_ok=True)
        source=args.source_root/design/'full-seed1-w300'/(design+'.pre-vpr.blif')
        row=dict(design=design,family=family,group=group,grid=[grid,grid],
                 source=str(source),source_available=source.is_file(),full_gp_or_route_executed=False)
        try:
            arch,blif=folder/'arch.xml',folder/'design.blif'
            if not arch.exists():
                write(folder/'architecture.json',generate_architecture(args.vtr_root/SOURCE_ARCH,arch,0,0,grid,grid))
            if not blif.exists():
                write(folder/'mapping.json',_module('prepare_natural_blif').prepare_blif(source,blif,arch))
            prefix=folder/'structure'
            trial=run_trial(folder/'structure_run',vpr_command(args.binary,arch,blif,32,701),
                inputs=[arch,blif],env=dict(TC_GP_STRUCTURE=prefix,TC_GP_STRUCTURE_ONLY=1),timeout=300,
                marker='TAILCOMPILE_V7_STRUCTURE_ONLY_COMPLETE',metadata=dict(binary_sha256=sha(args.binary)))
            if trial['status']!='complete':
                raise RuntimeError(f'structure export {trial["status"]}: {trial["log"]}')
            logic=build_logic(prefix,design,blif,arch,family=family,split='train')
            write(folder/'logic.json',logic)
            demand=[sum(c['demand'][a] for c in logic['clusters']) for a in range(5)]
            hard_ap_counts={kind:sum(c['resource']==kind for c in logic['clusters'])
                            for kind in ('memory','dsp_top')}
            hard_count=sum(hard_ap_counts.values())
            row.update(structure_compatible=True,node_count=len(logic['clusters']),demand=demand,
                       demand_keys=logic['demand_keys'],hard_ap_counts=hard_ap_counts,
                       demand_semantics=logic['demand_semantics'],heavy_light_threshold_passed=
                       hard_count>=16 if group=='heavy' else hard_count<=8)
        except Exception as exc:
            row.update(structure_compatible=False,error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
        write(args.out/'input_audit.json',dict(complete=len(rows)==len(cases),designs=rows,
            qualification_scope='pre-GP inputs only; candidate grids still require Default P&R'))
        print(row,flush=True)


if __name__=='__main__':
    main()
