#!/usr/bin/env python3
"""Auditable two-layer fixtures for the pinned VPR heterogeneity regression.

Only the layout of the official Hecate3D architecture is replaced.  All tile,
primitive, routing, switch, segment, CHANZ and scatter/gather definitions stay
unchanged.  ``single`` layout entries deliberately express every DSP/RAM site
explicitly, including singleton and missing-resource-layer cases.

This module is platform independent and never invokes VPR.  Its BLIF generator
generalizes the already tested two-lane circuit in probe_layer_heterogeneity.py;
RAMs remain bit sliced as required by the pinned VPR memory netlist model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


COMMIT = "40f8d19708b0b60e2aa04452deb1cfed68d96b0b"
SOURCE_ARCH = "vtr_flow/arch/multi_die/hecate/3d/hecate_3d_sb_10um_bidir_fanin_16_fanout_16_with_esd_7nm.xml"
KINDS = ("clb", "dsp_top", "memory", "io")


def _columns(xs: tuple[int, ...], height: int = 10) -> list[tuple[int, int]]:
    return [(x, y) for x in xs for y in range(1, height - 1)]


def _layer(dsp: list[tuple[int, int]], ram: list[tuple[int, int]]) -> dict:
    return {"dsp_top": dsp, "memory": ram}


CASES: dict[str, dict[str, Any]] = {
    "aligned": {
        "size": (10, 10), "lanes": 2,
        "layers": [_layer(_columns((2,)), _columns((5,))),
                   _layer(_columns((2,)), _columns((5,)))],
        "purpose": "Aligned two-layer control; the global tile definitions match every other case.",
    },
    "shifted": {
        "size": (10, 10), "lanes": 2,
        "layers": [_layer(_columns((2,)), _columns((5,))),
                   _layer(_columns((3,)), _columns((6,)))],
        "purpose": "All DSP and BRAM columns shift by one between layers; capacities stay equal.",
    },
    "single_site": {
        "size": (10, 10), "lanes": 2,
        "layers": [_layer([(2, 2)], [(5, 3)]), _layer([(3, 7)], [(6, 6)])],
        "purpose": "Exactly one compatible DSP and one BRAM site per layer, at different XY locations.",
        "targeted_checks": [
            "A cross-layer move must consider the target layer's only compatible site, even when its compressed index equals the source index.",
            "Two DSPs and two physical BRAMs require both layers, so a resource class cannot silently disappear from candidate search.",
        ],
    },
    "no_dsp_layer1": {
        "size": (10, 10), "lanes": 2,
        "layers": [_layer(_columns((2,)), _columns((5,))),
                   _layer([], _columns((6,)))],
        "purpose": "Upper layer has zero DSP sites; legal DSP moves must stay on the lower layer.",
    },
    "no_bram_layer0": {
        "size": (10, 10), "lanes": 2,
        "layers": [_layer(_columns((2,)), []),
                   _layer(_columns((3,)), _columns((6,)))],
        "purpose": "Lower layer has zero BRAM sites; legal BRAM moves must stay on the upper layer.",
    },
    "asymmetric": {
        "size": (10, 10), "lanes": 2,
        "layers": [_layer(_columns((2, 4)), [(5, 3)]),
                   _layer([(3, 6)], _columns((6, 8)))],
        "purpose": "DSP counts are 16:1 and BRAM counts 1:16; a type can have sparse versus dense target grids.",
    },
    "sparse": {
        "size": (10, 10), "lanes": 2,
        "layers": [
            _layer([(2, 1), (2, 5), (6, 3), (6, 7)],
                   [(4, 2), (4, 6), (8, 4), (8, 8)]),
            _layer([(3, 2), (3, 6), (7, 4), (7, 8)],
                   [(1, 3), (1, 7), (5, 1), (5, 5)]),
        ],
        "purpose": "Nonaligned sparse rows and columns exercise per-column Y lookup rather than rectangular compressed-grid assumptions.",
    },
    "larger": {
        "size": (40, 40), "lanes": 128,
        "layers": [_layer(_columns((2, 10, 18, 26), 40), _columns((5,), 40)),
                   _layer(_columns((7,), 40), _columns((6, 14, 22, 30), 40))],
        "purpose": "128-lane, 12315-atom scale-up with unequal 152:38 DSP and 38:152 BRAM capacity, and shifted hard-resource columns.",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(element: ET.Element) -> tuple:
    """Compare semantics without making indentation a primitive/routing change."""
    return (element.tag, tuple(sorted(element.attrib.items())),
            (element.text or "").strip(), tuple(_signature(c) for c in element))


def _nonlayout_signature(root: ET.Element) -> tuple:
    return tuple(_signature(c) for c in root if c.tag not in ("layout", "fixed_layout"))


def _tile_capacities(root: ET.Element) -> dict[str, int]:
    capacities = {}
    for kind in KINDS:
        tile = root.find(f"./tiles/tile[@name='{kind}']")
        if tile is None:
            raise ValueError(f"source architecture is missing tile {kind!r}")
        if int(tile.get("width", "1")) != 1 or int(tile.get("height", "1")) != 1:
            raise ValueError(f"fixture expects the official 1x1 {kind} tile")
        subtiles = tile.findall("sub_tile")
        if len(subtiles) != 1:
            raise ValueError(f"fixture expects the official single sub_tile definition for {kind}")
        capacities[kind] = int(subtiles[0].get("capacity", "1"))
    # This BLIF maps one full-width multiply/RAM to one hard-block slot.  Fail
    # clearly if a different source arch would make the audited counts invalid.
    if any(capacities[kind] != 1 for kind in ("clb", "dsp_top", "memory")):
        raise ValueError("fixture requires unit-capacity CLB/DSP/BRAM tiles")
    if capacities["io"] < 1:
        raise ValueError("source architecture IO capacity must be positive")
    return capacities


def _slots(spec: dict, capacities: dict[str, int]) -> list[dict]:
    """Enumerate actual physical slots as [x, y, subtile, layer]."""
    width, height = spec["size"]
    rows = []
    for layer, hard_sites in enumerate(spec["layers"]):
        interior = {(x, y) for x in range(1, width - 1) for y in range(1, height - 1)}
        dsp, ram = set(hard_sites["dsp_top"]), set(hard_sites["memory"])
        if len(dsp) != len(hard_sites["dsp_top"]) or len(ram) != len(hard_sites["memory"]):
            raise ValueError("duplicate hard-block site in fixture")
        if dsp & ram or not (dsp | ram) <= interior:
            raise ValueError("hard-block sites must be distinct interior coordinates")
        points = {"clb": interior - dsp - ram, "dsp_top": dsp, "memory": ram}
        points["io"] = {(x, y) for x in range(width) for y in range(height)
                        if (x in (0, width - 1)) != (y in (0, height - 1))}
        sites = {kind: [[x, y, sub, layer] for x, y in sorted(points[kind])
                        for sub in range(capacities[kind])] for kind in KINDS}
        rows.append({"layer": layer, "capacity": {kind: len(sites[kind]) for kind in KINDS},
                     "site_count": {kind: len(points[kind]) for kind in KINDS}, "slots": sites})
    return rows


def _write_blif(out: Path, lanes: int) -> None:
    inputs = ["clk", "we"] + [f"addr{i}" for i in range(9)]
    inputs += [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)]
    lines = [".model tiny", ".inputs " + " ".join(inputs),
             ".outputs " + " ".join(f"y{i}" for i in range(lanes))]
    for lane in range(lanes):
        mult = [f"a[{i}]=a{i % 8}" for i in range(27)]
        mult += [f"b[{i}]=b{i % 8}" for i in range(27)]
        mult += [f"out[{i}]=p{lane}_{i}" for i in range(54)]
        lines.append(".subckt multiply " + " ".join(mult))
        for bit in range(40):
            ram = [f"addr[{i}]=addr{i}" for i in range(9)]
            ram += [f"data=p{lane}_{bit}", "we=we", "clk=clk", f"out=q{lane}_{bit}"]
            lines.append(".subckt single_port_ram " + " ".join(ram))
        previous = f"q{lane}_0"
        values = [f"q{lane}_{i}" for i in range(1, 40)] + [f"p{lane}_{i}" for i in range(40, 54)]
        for index, value in enumerate(values):
            nxt = f"xor{lane}_{index}"
            lines.extend([f".names {previous} {value} {nxt}", "01 1", "10 1"])
            previous = nxt
        lines.append(f".latch {previous} y{lane} re clk 0")
    lines.append(".end")
    for model, inports, outports in (
        ("multiply", [f"a[{i}]" for i in range(27)] + [f"b[{i}]" for i in range(27)],
         [f"out[{i}]" for i in range(54)]),
        ("single_port_ram", [f"addr[{i}]" for i in range(9)] + ["data", "we", "clk"], ["out"]),
    ):
        lines.extend([f".model {model}", ".inputs " + " ".join(inports),
                      ".outputs " + " ".join(outports), ".blackbox", ".end"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_case(source_arch: str | Path, out_dir: str | Path, case_name: str) -> dict:
    """Write one fixture and return ``arch``, ``blif``, ``metadata`` and metadata_path.

    ``arch`` and ``blif`` are Path objects. ``metadata`` is the JSON-compatible
    audit dictionary, also saved to ``metadata_path``. Existing named artifacts
    are never overwritten; unrelated content in out_dir is left alone.
    """
    if case_name not in CASES:
        raise ValueError(f"unknown case {case_name!r}; choose from {', '.join(CASES)}")
    source_arch, out_dir = Path(source_arch), Path(out_dir)
    spec = copy.deepcopy(CASES[case_name])
    tree = ET.parse(source_arch)
    root = tree.getroot()
    signature = _nonlayout_signature(root)
    capacities = _tile_capacities(root)
    layers = _slots(spec, capacities)
    totals = {kind: sum(row["capacity"][kind] for row in layers) for kind in KINDS}
    lanes = spec["lanes"]
    if totals["dsp_top"] < lanes or totals["memory"] < lanes or totals["io"] < 27 + lanes:
        raise ValueError("fixture has insufficient physical hard-block or IO capacity")
    # The tested two-lane netlist packs six CLBs. Reserve a conservative six
    # CLBs/lane (including fragmentation headroom), not a claim about pack output.
    if totals["clb"] < 6 * lanes:
        raise ValueError("fixture lacks conservative CLB packing headroom")
    out_dir.mkdir(parents=True, exist_ok=True)
    arch, blif, metadata_path = out_dir / "arch.xml", out_dir / "tiny.blif", out_dir / "fixture.json"
    for target in (arch, blif, metadata_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing fixture artifact: {target}")
    for child in list(root):
        if child.tag in ("layout", "fixed_layout"):
            root.remove(child)
    width, height = spec["size"]
    layout = ET.Element("layout")
    fixed = ET.SubElement(layout, "fixed_layout", name="probe", width=str(width), height=str(height))
    for layer, hard_sites in enumerate(spec["layers"]):
        die = ET.SubElement(fixed, "layer", die=str(layer))
        ET.SubElement(die, "perimeter", type="io", priority="100")
        ET.SubElement(die, "corners", type="EMPTY", priority="101")
        ET.SubElement(die, "fill", type="clb", priority="10")
        for kind in ("dsp_top", "memory"):
            for x, y in sorted(hard_sites[kind]):
                ET.SubElement(die, "single", type=kind, priority="20", x=str(x), y=str(y))
    root.insert(2, layout)
    if _nonlayout_signature(root) != signature:
        raise AssertionError("fixture unexpectedly changed non-layout architecture definitions")
    ET.indent(tree, space="  ")
    tree.write(arch, encoding="utf-8", xml_declaration=True)
    _write_blif(blif, lanes)
    if _nonlayout_signature(ET.parse(arch).getroot()) != signature:
        raise AssertionError("serialized fixture changed non-layout architecture definitions")
    metadata = {
        "schema": "tailcompile.heterogeneity_fixture.v1",
        "case": case_name, "purpose": spec["purpose"], "vtr_commit": COMMIT,
        "source_arch": str(source_arch), "source_arch_sha256": _sha(source_arch),
        "architecture": arch.name, "architecture_sha256": _sha(arch),
        "blif": blif.name, "blif_sha256": _sha(blif),
        "layout_name": "probe", "width": width, "height": height, "layer_count": 2,
        "global_tile_and_primitive_definitions_unchanged": True,
        "routing_definitions_unchanged": True, "slot_order": ["x", "y", "subtile", "layer"],
        "tile_subtile_capacities": capacities, "layers": layers, "capacity_total": totals,
        "resource_available_layers": {kind: [row["layer"] for row in layers if row["capacity"][kind]]
                                      for kind in KINDS},
        "circuit": {
            "model": "tiny", "lanes": lanes,
            "atoms_before_vpr_optimization": 27 + 96 * lanes,
            "atom_counts": {"input": 27, "output": lanes, "multiply": lanes,
                            "single_port_ram_bit_slice": 40 * lanes, "xor_lut": 53 * lanes,
                            "latch": lanes},
            "expected_packed_hard_blocks": {"dsp_top": lanes, "memory": lanes},
            "estimated_clb_clusters": 3 * lanes,
            "estimated_packed_cluster_total": 27 + 6 * lanes,
            "packing_estimate_note": "The original two-lane case packs 2 DSP, 2 BRAM, 6 CLB, 29 IO. Hard-block counts are expected from full-width modes; CLB/cluster estimates must be replaced by observed pack counts.",
        },
        "recommended_matrix": {
            "placement_seeds": [11] if case_name == "larger" else [11, 12, 13],
            "followup_seeds": [12, 13] if case_name == "larger" else [],
            "routing_channel_widths": [320],
            "modes": ["initial_placement", "bounding_box_anneal", "timing_driven_anneal", "route_given_legal_place"],
            "routing_width_note": "Width 320 is a generous support smoke-test point, not a minimum-width or QoR claim; ordinary routing failure is not evidence of a layout-support bug.",
        },
        "targeted_checks": spec.get("targeted_checks", []),
        "acceptance": {
            "support": "Packing and placement initialize without invalid grid assumptions; every placed cluster occupies a compatible unused slot; missing target-layer resources yield no move, not a crash.",
            "routing": "Distinguish normal unroutable/congestion/timeout from assertion, invalid coordinate, resource mismatch, or RR-graph construction failure. A legal placement need not route at every chosen width.",
            "scale": "The larger fixture tests stability on the same repeated lane topology, not generalization to an independent real design or production-scale timing/QoR.",
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"arch": arch, "blif": blif, "metadata": metadata, "metadata_path": metadata_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list case names and purposes")
    parser.add_argument("--source-arch", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    args = parser.parse_args()
    if args.list:
        for name, spec in CASES.items():
            print(f"{name}: {spec['purpose']}")
        return 0
    if args.source_arch is None or args.out is None:
        parser.error("--source-arch and --out are required unless --list is used")
    for name in CASES if args.case == "all" else [args.case]:
        result = generate_case(args.source_arch, args.out / name, name)
        print(json.dumps({"case": name, "arch": str(result["arch"]), "blif": str(result["blif"]),
                          "capacity": result["metadata"]["capacity_total"],
                          "atoms": result["metadata"]["circuit"]["atoms_before_vpr_optimization"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
