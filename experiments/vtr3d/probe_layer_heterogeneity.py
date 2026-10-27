#!/usr/bin/env python3
"""Small, version-pinned diagnostic of aligned versus staggered 3D resources.

Run inside WSL. Generates two 10x10x2 architectures from the official Hecate
3D architecture, and a small BLIF exercising CLBs, DSPs and block memories.
The existing VTR checkout is read only. All outputs go into --out.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import resource
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

COMMIT = "40f8d19708b0b60e2aa04452deb1cfed68d96b0b"
SOURCE_ARCH = "vtr_flow/arch/multi_die/hecate/3d/hecate_3d_sb_10um_bidir_fanin_16_fanout_16_with_esd_7nm.xml"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def architecture(source, out, shifted):
    tree = ET.parse(source)
    root = tree.getroot()
    for child in list(root):
        if child.tag in ("layout", "fixed_layout"):
            root.remove(child)
    layout = ET.Element("layout")
    fixed = ET.SubElement(layout, "fixed_layout", name="probe", width="10", height="10")
    columns = []
    for layer in (0, 1):
        die = ET.SubElement(fixed, "layer", die=str(layer))
        ET.SubElement(die, "perimeter", type="io", priority="100")
        ET.SubElement(die, "corners", type="EMPTY", priority="101")
        ET.SubElement(die, "fill", type="clb", priority="10")
        dsp, memory = (3, 6) if shifted and layer == 1 else (2, 5)
        for kind, x in (("dsp_top", dsp), ("memory", memory)):
            ET.SubElement(die, "col", type=kind, startx=str(x), starty="1", priority="20")
        columns.append({"layer": layer, "dsp_column": dsp, "memory_column": memory})
    root.insert(2, layout)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return columns


def circuit(out):
    inputs = ["clk", "we"] + [f"addr{i}" for i in range(9)]
    inputs += [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)]
    lines = [".model tiny", ".inputs " + " ".join(inputs), ".outputs y0 y1"]
    for lane in (0, 1):
        mult = [f"a[{i}]=a{i % 8}" for i in range(27)]
        mult += [f"b[{i}]=b{i % 8}" for i in range(27)]
        mult += [f"out[{i}]=p{lane}_{i}" for i in range(54)]
        lines.append(".subckt multiply " + " ".join(mult))
        # VPR's memory-class netlist model is bit sliced; the packer combines
        # equal address/control slices into the physical 512x40 RAM mode.
        for bit in range(40):
            ram = [f"addr[{i}]=addr{i}" for i in range(9)]
            ram += [f"data=p{lane}_{bit}", "we=we", "clk=clk", f"out=q{lane}_{bit}"]
            lines.append(".subckt single_port_ram " + " ".join(ram))
        prev = f"q{lane}_0"
        # Also use every multiplier output to prevent dangling-output pruning.
        values = [f"q{lane}_{i}" for i in range(1, 40)] + [f"p{lane}_{i}" for i in range(40, 54)]
        for index, value in enumerate(values):
            nxt = f"xor{lane}_{index}"
            lines.extend([f".names {prev} {value} {nxt}", "01 1", "10 1"])
            prev = nxt
        lines.append(f".latch {prev} y{lane} re clk 0")
    lines.append(".end")
    for model, inports, outports in (
        ("multiply", [f"a[{i}]" for i in range(27)] + [f"b[{i}]" for i in range(27)], [f"out[{i}]" for i in range(54)]),
        ("single_port_ram", [f"addr[{i}]" for i in range(9)] + ["data", "we", "clk"], ["out"]),
    ):
        lines.extend([f".model {model}", ".inputs " + " ".join(inports), ".outputs " + " ".join(outports), ".blackbox", ".end"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def run(command, cwd, timeout):
    start = time.monotonic()
    try:
        result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout, preexec_fn=no_core)
        output, code = result.stdout, result.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or b""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        code = "timeout"
    (cwd / "console.log").write_text(output, encoding="utf-8")
    row = {"command": list(map(str, command)), "returncode": code,
           "elapsed_seconds": round(time.monotonic() - start, 4),
           "vpr_succeeded": "VPR succeeded" in output,
           "routed": "Circuit successfully routed" in output,
           "assertion": next((line for line in output.splitlines() if "Assertion" in line or "Assertion failure" in line), None),
           "error_tail": output.splitlines()[-22:] if code != 0 else []}
    row["initial_placement_succeeded"] = "Initial placement iteration 0 has finished successfully" in output
    cpd = re.search(r"Final critical path delay.*?:\s*([\d.eE+-]+) ns", output)
    row["final_cpd_ns"] = float(cpd.group(1)) if cpd else None
    route = cwd / "tiny.route"
    if route.is_file():
        route_text = route.read_text()
        ids = re.findall(r"Node:\s*(\d+)\s+CHANZ\b", route_text)
        row["routed_chanz_occurrences"] = len(ids)
        row["routed_unique_chanz_nodes"] = len(set(ids))
    save(cwd / "manifest.json", row)
    print(f"{cwd.parent.name}/{cwd.name}: code={code} routed={row['routed']} success={row['vpr_succeeded']}", flush=True)
    return row


def manual_place(net, out, columns):
    root = ET.parse(net).getroot()
    slots = collections.defaultdict(list)
    for row in columns:
        layer, dx, mx = row["layer"], row["dsp_column"], row["memory_column"]
        for x in range(1, 9):
            for y in range(1, 9):
                kind = "dsp_top" if x == dx else "memory" if x == mx else "clb"
                slots[kind, layer].append((x, y, 0, layer))
        for x in range(10):
            for y in range(10):
                if (x in (0, 9)) != (y in (0, 9)):
                    slots["io", layer].extend((x, y, sub, layer) for sub in range(8))
    counts = collections.Counter()
    audit, lines = [], [f"Netlist_File: {net.resolve()} Netlist_ID: SHA256:{sha(net)}", "Array size: 10 x 10 logic blocks", "", "#block name\tx\ty\tsubblk\tlayer\tblock number"]
    for index, block in enumerate(root.findall("block")):
        kind = block.attrib["instance"].split("[")[0]
        if block.attrib["name"] == "open":
            continue
        layer = counts[kind] % 2
        # The two full-size memories and multipliers alternate opposite layers.
        if kind == "memory":
            layer = 1 - layer
        counts[kind] += 1
        pos = slots[kind, layer].pop(0)
        lines.append(block.attrib["name"] + "\t" + "\t".join(map(str, pos)) + f"\t#{index}")
        audit.append({"name": block.attrib["name"], "type": kind, "location": pos})
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save(out.with_suffix(".audit.json"), audit)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vtr-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--width", type=int, default=320)
    a = p.parse_args()
    actual = subprocess.check_output(["git", "-C", str(a.vtr_root), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        p.error(f"expected VTR {COMMIT}, actual {actual}")
    if a.out.exists():
        p.error(f"output already exists: {a.out}")
    a.out.mkdir(parents=True)
    a.out = a.out.resolve()
    binary = a.vtr_root / "vpr/vpr"
    source = a.vtr_root / SOURCE_ARCH
    (a.out / "official_source_arch.xml").write_bytes(source.read_bytes())
    circuit(a.out / "tiny.blif")
    changes = subprocess.check_output(["git", "-C", str(a.vtr_root), "diff", "--ignore-submodules", "HEAD"], text=True)
    (a.out / "vtr_tracked_source_diff.patch").write_text(changes, encoding="utf-8")
    report = {"vtr_commit": actual, "binary": str(binary), "binary_sha256": sha(binary),
              "version_output": subprocess.check_output([str(binary), "--version"], text=True),
              "source_arch_sha256": sha(source), "source_diff_recorded": bool(changes),
              "source_diff_note": "Existing AP hook changes are recorded; explicit --pack/--place/--route stages do not execute the AP flow.",
              "grid": [10, 10, 2], "width": a.width, "seed": 1, "cases": {}}
    for variant in ("aligned", "shifted"):
        folder = a.out / variant
        folder.mkdir()
        arch = folder / "arch.xml"
        cols = architecture(source, arch, variant == "shifted")
        base = [str(binary), str(arch), str(a.out / "tiny.blif"), "--device", "probe", "--seed", "1", "--route_chan_width", str(a.width)]
        rows = {"resource_columns": cols, "arch_sha256": sha(arch)}
        for stage, flags in (("pack", ["--pack"]), ("full", ["--pack", "--place", "--route", "--analysis"])):
            work = folder / stage
            work.mkdir()
            rows[stage] = run(base + flags, work, a.timeout)
        pack_net = folder / "pack/tiny.net"
        if rows["pack"]["vpr_succeeded"] and pack_net.is_file():
            work = folder / "manual_route"
            work.mkdir()
            shutil.copy2(pack_net, work / "tiny.net")
            assignments = manual_place(work / "tiny.net", work / "tiny.place", cols)
            rows["manual_resource_counts"] = dict(collections.Counter(row["type"] for row in assignments))
            rows["manual_route"] = run(base + ["--route", "--analysis"], work, a.timeout)
        report["cases"][variant] = rows
        save(a.out / "summary.json", report)
    print(a.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
