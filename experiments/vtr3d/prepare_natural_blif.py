#!/usr/bin/env python3
"""Audited full-adder-to-LUT adaptation of existing natural-design BLIFs.

Only the pinned VTR one-bit adder is lowered. Other cells and all .names/.latch
records are retained; unknown models and ports fail before writing an output.
This is a derived technology mapping, not a new design or a full synthesis run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from heterogeneity_fixtures import COMMIT


ADDER_INPUTS = {"a", "b", "cin"}
ADDER_OUTPUTS = {"sumout", "cout"}
UNCONNECTED_OUTPUTS = {"unconn", "open"}
PIN = re.compile(r"([^\[\]\s=]+)(?:\[(\d+)\])?\Z")
PRIMITIVE_BASIS = {
    "vtr_commit": COMMIT,
    "path": "vtr_flow/primitives.v",
    "lines": [201, 219],
    "file_sha256": "e84a3a239a7f92cc8a66ef68cecc20bd714f979bb14b61d44bfd893be964b776",
    "definition": "module adder #(parameter WIDTH=1); assign {cout,sumout}=a+b+cin;",
}
SUPPORTED_DIRECTIVES = {".model", ".end", ".inputs", ".outputs", ".names", ".latch",
                        ".subckt", ".blackbox", ".clock"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _records(text):
    records, raw = [], ""
    for line in text.splitlines(keepends=True):
        raw += line
        content = line.split("#", 1)[0].rstrip()
        if content.endswith("\\"):
            continue
        combined = re.sub(r"\\\r?\n", " ", raw)
        tokens = " ".join(part.split("#", 1)[0] for part in combined.splitlines()).split()
        records.append((raw, tokens))
        raw = ""
    if raw:
        raise ValueError("unterminated BLIF continuation")
    return records


def _architecture_ports(path):
    root = ET.parse(path).getroot()
    models = {}
    for model in root.findall("models/model"):
        name = model.attrib["name"]
        if name in models:
            raise ValueError(f"duplicate architecture model {name}")
        inputs = {port.attrib["name"] for port in model.findall("input_ports/port")}
        outputs = {port.attrib["name"] for port in model.findall("output_ports/port")}
        models[name] = {"inputs": inputs, "outputs": outputs, "widths": {}}
    for block in root.iter("pb_type"):
        mapping = block.get("blif_model", "").split()
        if len(mapping) != 2 or mapping[0] != ".subckt" or mapping[1] not in models:
            continue
        widths = models[mapping[1]]["widths"]
        for tag in ("input", "output", "clock"):
            for port in block.findall(tag):
                name = port.attrib["name"]
                widths[name] = max(widths.get(name, 0), int(port.get("num_pins", "1")))
    return models


def _valid_arch_port(model, port, models, role=None):
    match = PIN.fullmatch(port)
    if match is None:
        raise ValueError(f"invalid BLIF port {model}.{port}")
    base, index = match.groups()
    spec = models[model]
    allowed = spec[role] if role else spec["inputs"] | spec["outputs"]
    if base not in allowed:
        raise ValueError(f"unknown {role or 'cell'} port {model}.{port}")
    if index is not None and int(index) >= spec["widths"].get(base, 0):
        raise ValueError(f"port index exceeds architecture maximum: {model}.{port}")


def adder_luts(pins):
    """Two exact 3-input truth tables; retain original net names, including inputs."""
    if set(pins) != ADDER_INPUTS | ADDER_OUTPUTS:
        raise ValueError("only scalar a,b,cin -> cout,sumout full adders are supported")
    if any(not value for value in pins.values()):
        raise ValueError("empty full-adder pin mapping")
    inputs = " ".join(pins[name] for name in ("a", "b", "cin"))
    lines = []
    if pins["sumout"] not in UNCONNECTED_OUTPUTS:
        lines.extend([f".names {inputs} {pins['sumout']}", "001 1", "010 1", "100 1", "111 1"])
    if pins["cout"] not in UNCONNECTED_OUTPUTS:
        lines.extend([f".names {inputs} {pins['cout']}", "11- 1", "1-1 1", "-11 1"])
    return "\n".join(lines) + ("\n" if lines else "")


def prepare_blif(source, out, architecture):
    source, out, architecture = Path(source), Path(out), Path(architecture)
    if source.resolve() == out.resolve():
        raise ValueError("derived BLIF must not overwrite its source")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite BLIF: {out}")
    original_hash = sha(source)
    with source.open(encoding="utf-8", newline="") as stream:
        text = stream.read()
    records, models = _records(text), _architecture_ports(architecture)
    declarations, active = {}, None
    for i, (_, tokens) in enumerate(records):
        if not tokens:
            continue
        command = tokens[0]
        if command.startswith(".") and command not in SUPPORTED_DIRECTIVES:
            raise ValueError(f"unsupported BLIF directive: {command}")
        if command == ".model":
            if active is not None or len(tokens) != 2 or tokens[1] in declarations:
                raise ValueError("malformed, nested or duplicate BLIF model")
            active = tokens[1]
            declarations[active] = {"start": i, "end": None, "inputs": set(), "outputs": set(),
                                    "blackbox": False, "commands": []}
        elif command == ".end":
            if active is None:
                raise ValueError("BLIF end outside model")
            declarations[active]["end"] = i
            active = None
        elif active is not None and command.startswith("."):
            declarations[active]["commands"].append(command)
            if command in (".inputs", ".outputs"):
                declarations[active][command[1:]].update(tokens[1:])
            elif command == ".blackbox":
                declarations[active]["blackbox"] = True
        elif active is None and command.startswith("."):
            raise ValueError(f"BLIF command outside model: {command}")
    if active is not None or not declarations:
        raise ValueError("BLIF has no complete model")
    top = next(iter(declarations))
    for name, declaration in declarations.items():
        if name == top:
            if declaration["blackbox"]:
                raise ValueError("top model is a blackbox")
            continue
        if name == "adder":
            if (declaration["inputs"] != ADDER_INPUTS or declaration["outputs"] != ADDER_OUTPUTS
                    or not declaration["blackbox"]
                    or set(declaration["commands"]) - {".inputs", ".outputs", ".blackbox"}):
                raise ValueError("adder model is not the known scalar VTR blackbox interface")
        elif name not in models:
            raise ValueError(f"unknown architecture model: {name}")
        else:
            if not declaration["blackbox"]:
                raise ValueError(f"non-blackbox hierarchy is outside this adapter: {name}")
            for role in ("inputs", "outputs"):
                for port in declaration[role]:
                    _valid_arch_port(name, port, models, role)
    pieces, counts, skipped_outputs = [], Counter(), Counter()
    omitted, active = set(), None
    if "adder" in declarations and "adder" != top:
        d = declarations["adder"]
        omitted.update(range(d["start"], d["end"] + 1))
    for i, (raw, tokens) in enumerate(records):
        if i in omitted:
            continue
        if tokens and tokens[0] == ".model":
            active = tokens[1]
        if tokens and tokens[0] == ".subckt":
            if active != top:
                raise ValueError("only flat top-level cell instances are supported")
            if len(tokens) < 3:
                raise ValueError("malformed cell instance")
            name, pins = tokens[1], {}
            for item in tokens[2:]:
                if "=" not in item:
                    raise ValueError(f"malformed cell pin: {item}")
                port, net = item.split("=", 1)
                if port in pins or not port or not net:
                    raise ValueError(f"duplicate or empty pin on {name}: {item}")
                pins[port] = net
            counts[name] += 1
            if name == "adder":
                if "adder" not in declarations:
                    raise ValueError("adder instance lacks its auditable scalar blackbox declaration")
                for port in ADDER_OUTPUTS:
                    if pins.get(port) in UNCONNECTED_OUTPUTS:
                        skipped_outputs[port] += 1
                pieces.append(adder_luts(pins))
                continue
            if name not in models:
                raise ValueError(f"unknown architecture cell instance: {name}")
            for port in pins:
                _valid_arch_port(name, port, models)
        pieces.append(raw)
    derived = "".join(pieces)
    if sha(source) != original_hash:
        raise AssertionError("source changed while preparing derived mapping")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8", newline="") as stream:
        stream.write(derived)
    return {
        "schema": "tailcompile-natural-blif-adaptation-v1", "top_model": top,
        "source": str(source), "source_sha256": original_hash,
        "derived": str(out), "derived_sha256": sha(out),
        "architecture": str(architecture), "architecture_sha256": sha(architecture),
        "source_unchanged": sha(source) == original_hash,
        "replaced_adder_count": counts["adder"],
        "removed_unused_adder_model_count": int("adder" in declarations and "adder" != top),
        "unconnected_adder_outputs_omitted": dict(skipped_outputs),
        "cell_counts_before": dict(sorted(counts.items())),
        "retained_cell_counts": {k: v for k, v in sorted(counts.items()) if k != "adder"},
        "mapping": {"inputs": ["a", "b", "cin"], "sumout": "a XOR b XOR cin",
                    "cout": "majority(a,b,cin)", "lut_count_per_fully_connected_adder": 2,
                    "basis": PRIMITIVE_BASIS},
        "equivalence_scope": "exhaustive one-bit truth-function proof; compositional replacement with identical net names, not an RTL re-synthesis or whole-design formal proof",
        "unconnected_input_policy": "retain the original input net name; no implicit tie-off or fresh input is introduced",
        "identity_note": "derived technology mapping of the same natural design; do not count it as an independent design family",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--architecture", type=Path, required=True)
    p.add_argument("--audit", type=Path)
    a = p.parse_args()
    if a.audit and a.audit.exists():
        p.error("audit output already exists")
    if a.audit and a.audit.resolve() in {a.source.resolve(), a.out.resolve(), a.architecture.resolve()}:
        p.error("audit must use a separate path from source, derived BLIF and architecture")
    result = prepare_blif(a.source, a.out, a.architecture)
    if a.audit:
        a.audit.parent.mkdir(parents=True, exist_ok=True)
        a.audit.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
