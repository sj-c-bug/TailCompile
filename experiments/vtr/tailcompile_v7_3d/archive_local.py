"""Archive light evidence from this campaign; no large payloads or deletions."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
from pathlib import Path
import shutil
from io_utils import read, write, sha


def compact(value):
    if isinstance(value, dict):
        return {k:compact(v) for k,v in value.items() if k != 'positions'}
    if isinstance(value,list):
        return [compact(v) for v in value]
    return value


def archive(source,out):
    source,out = Path(source).resolve(),Path(out).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('archive source and destination must be disjoint directories')
    previous = out/'archive_manifest.json'
    if previous.is_file() and read(previous).get('source') != str(source):
        raise ValueError('destination already belongs to a different source archive')
    out.mkdir(parents=True,exist_ok=True)
    files, cleaned, bytes_cleaned = [],0,0
    for path in source.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source)
        wanted = (path.suffix == '.json' or path.name in ('vpr.out','console.log','resources.txt') or
                  (path.suffix == '.tsv' and ('exports' in path.parts or 'initializers' in path.parts)))
        if not wanted:
            continue
        target = out/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        original_hash = sha(path)
        if path.suffix == '.json':
            value=read(path)
            if path.name=='payload_pruned.json':
                bytes_cleaned += value.get('removed_bytes',0)
                cleaned += len(value.get('removed',[]))
            write(target,compact(value))
        elif path.suffix=='.tsv':
            target=target.with_suffix(target.suffix+'.gz')
            with path.open('rb') as src,gzip.open(target,'wb') as dst:
                shutil.copyfileobj(src,dst)
        else:
            shutil.copyfile(path,target)
        files.append(dict(source_relative=str(relative),source_sha256=original_hash,
                          archived_relative=str(target.relative_to(out)),archived_sha256=sha(target),
                          archived_bytes=target.stat().st_size))
    manifests=[read(p) for p in source.rglob('manifest.json') if 'signature' in read(p)]
    statuses=Counter(row.get('status','unknown') for row in manifests)
    result=dict(source=str(source),archived_file_count=len(files),files=files,
                trial_statuses=dict(statuses),completed_payload_files_removed=cleaned,
                completed_payload_bytes_removed=bytes_cleaned,
                source_payload_deletion_performed_here=False,
                qualification_positions_compacted=True,original_inputs_preserved=True)
    write(out/'archive_manifest.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=archive(args.source,args.out)
    print({key:value for key,value in result.items() if key!='files'})


if __name__=='__main__':
    main()
