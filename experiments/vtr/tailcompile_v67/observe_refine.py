"""Stage instrumentation around the unchanged, frozen refinement algorithm."""
import argparse
import json
import os
from pathlib import Path
import time


def emit(path, stage, **details):
    row=dict(stage=stage,unix_time=time.time(),pid=os.getpid(),**details)
    if stage=='worker_start' and hasattr(os,'sched_getaffinity'):
        row['allowed_cpus']=sorted(os.sched_getaffinity(0))
    with path.open('a',encoding='utf-8') as log:
        log.write(json.dumps(row)+'\n');log.flush()
    print('TAILCOMPILE_WORKER_STAGE '+json.dumps(row),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,required=True)
    a=p.parse_args();events=a.output.with_name('worker.events.jsonl')
    emit(events,'worker_start')
    import refine
    emit(events,'worker_import_complete')
    # No action, score, random seed or search budget changes.
    for name,stage in [('snapshot','snapshot'),('candidates','local_search'),('flat','fplace_output')]:
        original=getattr(refine,name)
        def observed(*args,_fn=original,_stage=stage,**kwargs):
            emit(events,_stage+'_start');start=time.monotonic()
            result=_fn(*args,**kwargs)
            emit(events,_stage+'_complete',seconds=time.monotonic()-start)
            return result
        setattr(refine,name,observed)
    try:
        refine.run(a.prefix,a.output,a.width)
    except BaseException as error:
        emit(events,'worker_error',error=repr(error));raise
    emit(events,'worker_complete')


if __name__=='__main__':main()
