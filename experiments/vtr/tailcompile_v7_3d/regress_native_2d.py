"""Four real old/new, cold/warm native-GP regressions without changing old VTR."""
import argparse
import json
from pathlib import Path

from io_utils import read, run_trial, sha, write, prune_derived


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-root',type=Path,required=True)
    p.add_argument('--new-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    old,new=a.old_root.resolve(),a.new_root.resolve()
    out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    arch=old/'vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml'
    blif=old/'vtr_flow/benchmarks/blif/4/5xp1.blif'
    before={str(old/'vpr/vpr'):sha(old/'vpr/vpr')}
    records={}
    def run(method,mode,prior=None):
        binary=(old if method=='old' else new)/'vpr/vpr'
        folder=out/f'{method}_{mode}'
        prefix=folder/'teacher'
        env={'TC_GP_EXPORT':prefix}
        if prior:
            env['TC_GP_INIT']=prior
        command=[binary,arch,blif,'--analytical_place','--route','--analysis',
                 '--ap_full_legalizer','appack','--ap_detailed_placer','annealer',
                 '--device_width','14','--route_chan_width','100','--num_workers','1','--seed','701']
        record=run_trial(folder,command,inputs=[binary,arch,blif]+([prior] if prior else []),
                         env=env,timeout=180,prune=False,metadata=dict(scope='single_layer_native_gp_regression'))
        if record['status']!='success':
            raise RuntimeError(f'2D regression failed: {record["log"]}')
        text=Path(record['log']).read_text()
        tag='TAILCOMPILE_V68_' if method=='old' else 'TAILCOMPILE_V7_'
        if prior and (tag+'GP_INIT_CONSUMED' not in text or tag+'NATIVE_SOLVER_CONTINUED iteration=1' not in text):
            raise AssertionError('cold/warm contract markers missing')
        rows=[line.split('\t') for line in Path(str(prefix)+'.blocks.tsv').read_text().splitlines()]
        records[f'{method}_{mode}']=record
        prune_derived(record['workdir'],protected=[arch,blif]+([prior] if prior else []))
        return rows
    cold_old=run('old','cold')
    cold_new=run('new','cold')
    prior=out/'shared_init.ap.tsv'
    prior.write_text(''.join(' '.join([row[1],*row[4:8]])+'\n' for row in cold_old if row[3]=='0'))
    warm_old=run('old','warm',prior)
    warm_new=run('new','warm',prior)
    def locations(rows):
        return {row[1]:tuple(map(float,row[4:8])) for row in rows}
    cold_equal=locations(cold_old)==locations(cold_new)
    warm_equal=locations(warm_old)==locations(warm_new)
    after={path:sha(path) for path in before}
    passed=cold_equal and warm_equal and before==after
    report=dict(status='passed' if passed else 'failed',cold_native_gp_identical=cold_equal,
                warm_native_gp_identical=warm_equal,old_binary_unchanged=before==after,
                original_before=before,original_after=after,trials=records,
                note='Identical raw native GP XYZ checked before APPack. SA trajectories may change under the documented v2 inclusive-search fix.')
    write(out/'summary.json',report)
    print(json.dumps({key:value for key,value in report.items() if key!='trials'}))
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
