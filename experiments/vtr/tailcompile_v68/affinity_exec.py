"""Apply/record affinity in the child, never in a Python scheduler thread."""
import os
from pathlib import Path
import sys
import time

from common import write


def main():
    args = sys.argv[1:]
    split = args.index('--')
    cpu = int(args[args.index('--cpu') + 1])
    audit = Path(args[args.index('--audit') + 1])
    os.sched_setaffinity(0, {cpu})
    actual = sorted(os.sched_getaffinity(0))
    if actual != [cpu]:
        raise RuntimeError('Requested physical-core affinity was not applied')
    write(audit, dict(pid=os.getpid(), requested_cpu=cpu, allowed_cpus=actual,
        unix_time=time.time(), thread_env={key: os.environ.get(key) for key in
        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}))
    print(f'TAILCOMPILE_TIMING_AFFINITY cpu={cpu} allowed={actual}', flush=True)
    command = args[split + 1:]
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
