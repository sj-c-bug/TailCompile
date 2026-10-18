"""Four real VPR trials to validate both refinement entry paths, not efficacy."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import os
import four_routes as f

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--blif',type=Path,required=True)
p.add_argument('--packed',type=Path,required=True)
p.add_argument('--fplace',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--width',type=int,default=100)
p.add_argument('--device-width',type=int,default=14)
a=p.parse_args();a.out=a.out.resolve()
a.out.mkdir(parents=True,exist_ok=True)
# Git checkouts on Windows may contain CRLF; VPR's .fplace parser requires LF.
normalized=a.out/'input_lf.fplace'
normalized.write_text(a.fplace.read_text(),newline='\n')
b=f.direct.DirectBackend(os.environ['VTR_ROOT'])
identity=dict(binary=b.binary_hash,hook=f.verify(Path(b.root)),code=f.code_identity())
c=dict(design='smoke',width=a.width,device_width=a.device_width,blif=str(a.blif.resolve()),
       packed=str(a.packed.resolve()),fplace=str(normalized),
       fplace_sha256=f.sha(normalized),sources=dict(mapped_blif=f.sha(a.blif),packed_net=f.sha(a.packed)))
rows=[]
for m in f.METHODS:
    rows.append(f.run_trial(SimpleNamespace(timeout=600),b,identity,(c,251,m,a.out/m)))
f.write(a.out/'smoke_report.json',dict(interface_only=True,rows=rows))
