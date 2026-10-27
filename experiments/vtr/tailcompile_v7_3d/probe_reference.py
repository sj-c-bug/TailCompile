"""Export one unchanged-architecture RRG and Default engineering reference."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from device_graph import build_device_graph
from io_utils import read, write, run_trial, vpr_command, prune_derived, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--architecture',type=Path,required=True)
    p.add_argument('--blif',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--width',type=int,default=320)
    a=p.parse_args()
    a.out=a.out.resolve()
    a.out.mkdir(parents=True,exist_ok=True)
    rrg=a.out/'rr_graph.xml'
    device=a.out/'device_graph.json'
    fixed=ET.parse(a.architecture).getroot().find('layout/fixed_layout')
    name=fixed.get('name')
    pack=run_trial(a.out/'pack',[a.binary,a.architecture,a.blif,'--pack','--device',name,
        '--num_workers','1','--seed','701','--route_chan_width',str(a.width),
        '--place_chan_width',str(a.width),'--write_rr_graph',rrg],
        inputs=[a.binary,a.architecture,a.blif],timeout=180,prune=False,
        metadata=dict(scope='unchanged_architecture_engineering_reference',width=a.width))
    if pack['status']!='complete' or not rrg.is_file():
        raise RuntimeError(f'engineering RRG export failed: {pack["log"]}')
    if not device.exists():
        graph=build_device_graph(rrg,a.architecture,tile_size=4)
        write(device,graph)
    graph=read(device)
    prune_derived(pack['workdir'],protected=[a.architecture,a.blif,a.binary,rrg,device])
    default=run_trial(a.out/'default',vpr_command(a.binary,a.architecture,a.blif,a.width,511,rr_graph=rrg),
        inputs=[a.binary,a.architecture,a.blif,rrg],timeout=180,
        metadata=dict(scope='unchanged_architecture_engineering_reference',width=a.width))
    report=dict(schema='tailcompile-v7-engineering-reference-1',architecture_sha256=sha(a.architecture),
        binary_sha256=sha(a.binary),width=a.width,device_graph=str(device),rr_graph=str(rrg),
        regions=len(graph['regions']),pools=len(graph['pools']),grid=graph['grid'],
        z_capacity=graph['audit']['unique_connected_z_capacity'],
        z_by_class=graph['audit']['z_capacity_by_class'],default_status=default['status'],
        default_metrics=default['metrics'],
        scope='Wider Default engineering reference; no optimization tuning or research-width substitution')
    write(a.out/'reference_report.json',report)
    print(json.dumps(report))


if __name__=='__main__':
    main()
