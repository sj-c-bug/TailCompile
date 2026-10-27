"""Reclaim one parsed W320 engineering RRG after the learned smoke completes."""
import argparse
from pathlib import Path
import shutil

from io_utils import read,sha,write

BASE=Path('/home/phoenix/vtr_work/tailcompile_3d/v7_smoke_graph_w320_20261024')
SMOKE=Path('/home/phoenix/vtr_work/tailcompile_3d/v7_learned_smoke_w320_20261024')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    target=BASE/'rr_graph.xml'
    graph_path=BASE/'device_graph.json'
    graph=read(graph_path)
    reference=read(BASE/'reference_report.json')
    protocol=read(SMOKE/'protocol.json')
    report=read(SMOKE/'smoke_report.json')
    if not report.get('interface_passed') or not report.get('complete'):
        raise ValueError('completed learned smoke required before dropping its source RRG')
    if target.is_symlink() or target.resolve()!=target or not target.is_file():
        raise ValueError('expected exact nonsymlink raw RRG missing')
    if sha(target)!=graph['provenance']['rr_graph_sha256']:
        raise ValueError('RRG provenance hash mismatch')
    if sha(graph_path)!=protocol['device_graph']:
        raise ValueError('downstream smoke graph hash mismatch')
    architecture=Path(graph['provenance']['architecture'])
    if sha(architecture)!=graph['provenance']['architecture_sha256'] or reference['architecture_sha256']!=sha(architecture):
        raise ValueError('architecture provenance mismatch')
    protected=[graph_path,architecture,BASE/'reference_report.json',SMOKE/'protocol.json',
               SMOKE/'smoke_report.json',Path(report['checkpoint'])]
    anchors={str(path):sha(path) for path in protected}
    item=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,
              allocated_bytes=target.stat().st_blocks*512)
    plan=dict(target=item,protected_sha256=anchors,
              dependency='completed learned smoke reads device_graph JSON; all source RRG consumers complete',
              recovery='rerun recorded pack-only command in BASE/pack/manifest.json')
    plan_path=a.out.with_name(a.out.stem+'_plan.json')
    if plan_path.exists() or a.out.exists():
        raise ValueError('new cleanup ledger required')
    write(plan_path,plan)
    before=shutil.disk_usage(BASE).free
    if sha(target)!=item['sha256']:
        raise ValueError('RRG changed after planning')
    target.unlink()
    if anchors!={str(path):sha(path) for path in protected}:
        raise AssertionError('preserved downstream evidence changed')
    ledger=dict(applied=True,removed=item,protected_unchanged=True,
                protected_sha256=anchors,filesystem_free_change_observed=shutil.disk_usage(BASE).free-before,
                raw_rrg_reanalysis_requires_regeneration=True,plan_sha256=sha(plan_path))
    write(a.out,ledger)
    print({key:value for key,value in ledger.items() if key!='protected_sha256'})


if __name__=='__main__':
    main()
