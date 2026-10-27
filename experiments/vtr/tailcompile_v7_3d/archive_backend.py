"""Archive completed backend evidence and reclaim proven derived route/net files."""
import argparse
import json
from pathlib import Path
import re
import shutil

from io_utils import read, sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    a=p.parse_args()
    source,archive=a.source.resolve(),a.archive.resolve()
    candidates=[source/'smoke_report.json',source/'summary.json']
    completed=next((path for path in candidates if path.is_file()),None)
    if completed is None:
        p.error('completed smoke_report.json or summary.json required')
    report=read(completed)
    if not (report.get('status')=='passed' or report.get('passed') is True):
        p.error('refuse cleanup of unfinished/failed backend qualification')
    if archive.exists() or source==archive or source in archive.parents or archive in source.parents:
        p.error('archive must be new and disjoint from source')
    protected=set()
    def examine(value):
        if isinstance(value,dict):
            command=value.get('command')
            if isinstance(command,list):
                for idx,item in enumerate(command[:-1]):
                    if item=='--net_file':
                        path=Path(command[idx+1])
                        if path.is_absolute():
                            protected.add(path.resolve())
            for item in value.values():
                examine(item)
        elif isinstance(value,list):
            for item in value:
                examine(item)
    examine(report)
    payload=[]
    route_evidence=[]
    for path in source.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.resolve() in protected:
            continue
        if path.name.endswith(('.net','.net.post_routing','.net_postrouting','.net.postrouting','.route')):
            if not path.resolve().is_relative_to(source):
                raise ValueError('derived payload escaped exact completed root')
            payload.append(dict(path=str(path.relative_to(source)),bytes=path.stat().st_size,sha256=sha(path)))
            if path.suffix=='.route':
                text=path.read_text(errors='replace')
                ids=sorted(set(map(int,re.findall(r'Node:\s*(\d+)\s+CHANZ\b',text))))
                route_evidence.append(dict(path=str(path.relative_to(source)),unique_chanz_nodes=len(ids),chanz_node_ids=ids))
    write(source/'route_resource_evidence.json',dict(routes=route_evidence,
          note='Unique CHANZ IDs from final route text; not independent net count or capacity utilization'))
    write(source/'backend_payload_plan.json',dict(completion_evidence=str(completed),
          protected_inputs=sorted(map(str,protected)),derived=payload,
          planned_bytes=sum(item['bytes'] for item in payload)))
    ignored={item['path'] for item in payload}
    def ignore(directory,names):
        return [name for name in names if str((Path(directory)/name).relative_to(source)) in ignored]
    shutil.copytree(source,archive,ignore=ignore)
    for item in payload:
        path=source/item['path']
        if sha(path)!=item['sha256']:
            raise ValueError('payload changed since cleanup plan; stopped')
        path.unlink()
    ledger=dict(removed=payload,removed_bytes=sum(item['bytes'] for item in payload),
                protected_inputs=sorted(map(str,protected)),archive=str(archive),
                raw_route_reanalysis_requires_rerun=bool(route_evidence),
                reason='qualification complete; compact metrics/logs/coordinates and CHANZ IDs archived before reclamation')
    write(source/'backend_payload_pruned.json',ledger)
    write(archive/'backend_payload_pruned.json',ledger)
    print(json.dumps(dict(archive=str(archive),removed_files=len(payload),removed_bytes=ledger['removed_bytes'])))


if __name__=='__main__':
    main()
