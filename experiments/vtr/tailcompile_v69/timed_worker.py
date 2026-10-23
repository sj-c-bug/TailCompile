"""Pin a full Default/initializer+VPR implementation; emit exact subprocess cost."""
import argparse
import os
from pathlib import Path
import subprocess
import time
from bridge import read, write


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    a=p.parse_args();spec=read(a.spec);folder=a.spec.parent
    os.sched_setaffinity(0,{spec['cpu']})
    allowed=sorted(os.sched_getaffinity(0))
    if allowed!=[spec['cpu']]:
        raise RuntimeError('CPU affinity not applied')
    write(folder/'affinity.json',dict(cpu=spec['cpu'],allowed_cpus=allowed,
        thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    env={k:v for k,v in os.environ.items() if not k.startswith('TC_GP_')}
    generation=None
    if spec['method']=='fast':
        from generate import generate
        generation=generate(spec['context'],spec['checkpoint'],folder/'initializer',
                            spec['options'],spec['candidate'],reuse=False)
        if generation['ap_init_sha256']!=spec['expected_ap_sha256']:
            raise ValueError('Timed inference differs from the frozen scan candidate')
        env['TC_GP_INIT']=generation['ap_init']
    # Native stdout remains in the parent's vpr.out. Parent process group owns
    # this worker AND VPR, so timeout/recovery cannot leave an untracked child.
    start=time.perf_counter()
    result=subprocess.run(spec['vpr_command'],env=env)
    elapsed=time.perf_counter()-start
    write(folder/'cost.json',dict(vpr_seconds=elapsed,returncode=result.returncode,
        generation=generation,rrg_encoding_seconds=generation['timing']['rrg_encoding'] if generation else 0.))
    raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
