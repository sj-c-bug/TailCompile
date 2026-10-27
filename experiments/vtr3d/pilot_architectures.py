#!/usr/bin/env python3
"""Generate resource-total-matched two-layer architectures for the 3D pilot.

This is a layout generator, not an architecture or primitive translator. The
same hard-column coordinates are used in both variants. Only their resource
types change: 2 DSP + 2 RAM per layer versus 3:1 / 1:3. All non-layout XML,
including the vertical-routing definitions, remains semantically unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from heterogeneity_fixtures import (COMMIT, KINDS, SOURCE_ARCH,
                                    _nonlayout_signature, _slots,
                                    _tile_capacities)


VARIANTS = ("homogeneous", "heterogeneous")


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate_architecture(source_arch, out, variant, width, height, hard_columns=4):
    """Write a fresh XML and return an auditable, JSON-compatible metadata dict.

    ``hard_columns`` is the total number of full columns of EACH hard-resource
    type across both layers, and also the number of occupied hard columns per
    layer. It must be even and >=4 so both variants contain DSP and RAM on both
    layers. Slot coordinates have order [x, y, subtile, layer].
    """
    source_arch, out = Path(source_arch), Path(out)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose {VARIANTS}")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (width, height, hard_columns)):
        raise ValueError("grid dimensions and hard_columns must be integers")
    if hard_columns < 4 or hard_columns % 2:
        raise ValueError("hard_columns must be even and >=4, with both types on both layers")
    if width < max(12, hard_columns + 4) or height < 6:
        raise ValueError("grid too small for distinct hard columns and CLB headroom")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite architecture: {out}")
    tree = ET.parse(source_arch)
    root = tree.getroot()
    signature = _nonlayout_signature(root)
    capacities = _tile_capacities(root)
    xs = [((i + 1) * (width - 1)) // (hard_columns + 1) for i in range(hard_columns)]
    if len(set(xs)) != hard_columns or min(xs) < 1 or max(xs) >= width - 1:
        raise AssertionError("hard-column placement is not distinct and interior")
    half = hard_columns // 2
    homogeneous_types = [kind for _ in range(half) for kind in ("dsp_top", "memory")]
    types = ([homogeneous_types, homogeneous_types] if variant == "homogeneous" else
             [["dsp_top"] * (hard_columns - 1) + ["memory"],
              ["dsp_top"] + ["memory"] * (hard_columns - 1)])
    spec = {"size": (width, height), "layers": []}
    for kinds in types:
        layer = {kind: [] for kind in ("dsp_top", "memory")}
        for x, kind in zip(xs, kinds):
            layer[kind].extend((x, y) for y in range(1, height - 1))
        spec["layers"].append(layer)
    slots = _slots(spec, capacities)
    totals = {kind: sum(layer["capacity"][kind] for layer in slots) for kind in KINDS}
    expected = {"dsp_top": hard_columns * (height - 2),
                "memory": hard_columns * (height - 2),
                "clb": 2 * (width - 2 - hard_columns) * (height - 2),
                "io": 2 * (2 * width + 2 * height - 8) * capacities["io"]}
    if totals != expected:
        raise AssertionError(f"layout capacity mismatch: {totals} != {expected}")
    for node in list(root):
        if node.tag in ("layout", "fixed_layout"):
            root.remove(node)
    layout = ET.Element("layout")
    fixed = ET.SubElement(layout, "fixed_layout", name="pilot", width=str(width), height=str(height))
    for z, kinds in enumerate(types):
        layer = ET.SubElement(fixed, "layer", die=str(z))
        ET.SubElement(layer, "perimeter", type="io", priority="100")
        ET.SubElement(layer, "corners", type="EMPTY", priority="101")
        ET.SubElement(layer, "fill", type="clb", priority="10")
        for x, kind in zip(xs, kinds):
            ET.SubElement(layer, "col", type=kind, startx=str(x), starty="1", priority="20")
    root.insert(2, layout)
    if _nonlayout_signature(root) != signature:
        raise AssertionError("non-layout architecture changed")
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    if _nonlayout_signature(ET.parse(out).getroot()) != signature:
        raise AssertionError("serialized non-layout architecture changed")
    return {
        "schema": "tailcompile-3d-matched-architecture-v1",
        "variant": variant, "device_name": "pilot", "vtr_commit": COMMIT,
        "architecture": str(out), "architecture_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "source_arch": str(source_arch), "source_arch_sha256": hashlib.sha256(source_arch.read_bytes()).hexdigest(),
        "nonlayout_definition_sha256": _hash(signature),
        "nonlayout_definitions_unchanged": True,
        "vertical_routing_definitions_unchanged": True,
        "width": width, "height": height, "layer_count": 2,
        "grid": {"width": width, "height": height, "layers": 2},
        "hard_columns_per_resource_total": hard_columns,
        "hard_column_x": xs, "hard_column_types_by_layer": types,
        "slot_order": ["x", "y", "subtile", "layer"],
        "layers": slots, "capacity_total": totals,
        "comparison_control": "same total typed resources, grid, hard-column coordinates, IO and all non-layout definitions",
        "limitations": ["active resources are the official unit-size CLB/DSP/RAM and perimeter IO",
                        "matching XML definitions does not certify detailed RR reachability or equal routing demand",
                        "width calibration and all P&R measurements must use the same backend and conditions"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-arch", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--hard-columns", type=int, default=4)
    a = p.parse_args()
    result = generate_architecture(a.source_arch, a.out, a.variant, a.width, a.height, a.hard_columns)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
