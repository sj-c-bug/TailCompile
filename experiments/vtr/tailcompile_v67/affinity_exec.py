"""Bind this child before exec; VPR and its refinement subprocess inherit it."""
import json
import os
from pathlib import Path
import sys
import time


def main():
    args=sys.argv[1:];split=args.index('--')
    cpu=int(args[args.index('--cpu')+1]);audit=Path(args[args.index('--audit')+1])
    os.sched_setaffinity(0,{cpu})
    actual=sorted(os.sched_getaffinity(0))
    if actual!=[cpu]:raise RuntimeError('CPU affinity was not applied')
    audit.write_text(json.dumps(dict(pid=os.getpid(),requested_cpu=cpu,allowed_cpus=actual,
        unix_time=time.time(),thread_env={k:os.environ.get(k) for k in
        ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))+'\n')
    print(f'TAILCOMPILE_CPU_AFFINITY cpu={cpu} allowed={actual}',flush=True)
    command=args[split+1:]
    os.execvpe(command[0],command,os.environ)


if __name__=='__main__':main()
