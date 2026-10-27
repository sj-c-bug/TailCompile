#!/usr/bin/env python3
"""Pinned end-to-end regression for same-XML, layer-heterogeneous VPR devices.

Run in Linux/WSL. Uses a fresh output directory, packs once per fixture, then
places/routes the frozen packed netlist with several placement seeds. It never
patches the existing VTR installation. This tests support, not QoR superiority.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET

from probe_layer_heterogeneity import COMMIT, SOURCE_ARCH, run, save, sha


def validate_place(net, place, metadata):
    """Independently check all cluster positions against generated physical slots."""
    blocks = {
        block.attrib["name"]: block.attrib["instance"].split("[")[0]
        for block in ET.parse(net).getroot().findall("block")
        if block.attrib["name"] != "open"
    }
    slots = {}
    # The generator documents its slots as [x, y, subtile, layer].
    for layer in metadata["layers"]:
        for kind, positions in layer["slots"].items():
            slots.setdefault(kind, set()).update(tuple(pos) for pos in positions)
    positions = {}
    errors = []
    occupied = {}
    for line in Path(place).read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if fields[0] not in blocks:
            # Header rows are not coordinates. Unknown coordinate rows are not
            # silently accepted as harmless headers.
            try:
                if len(fields) == 5:
                    tuple(map(int, fields[1:]))
                    errors.append(f"unknown cluster {fields[0]}")
            except ValueError:
                pass
            continue
        name = fields[0]
        if len(fields) != 5:
            errors.append(f"malformed coordinate row for {name}")
            continue
        try:
            pos = tuple(map(int, fields[1:]))
        except ValueError:
            errors.append(f"noninteger coordinate for {name}")
            continue
        if name in positions:
            errors.append(f"duplicate cluster {name}")
        positions[name] = pos
        if pos not in slots.get(blocks[name], set()):
            errors.append(f"incompatible or off-grid {name}: {blocks[name]} at {pos}")
        if pos in occupied:
            errors.append(f"collision {name} and {occupied[pos]} at {pos}")
        occupied[pos] = name
    missing = sorted(set(blocks) - set(positions))
    if missing:
        errors.append(f"missing {len(missing)} clusters: {missing[:8]}")
    counts = Counter(f"{blocks[name]}@{pos[3]}" for name, pos in positions.items())
    return {"passed": not errors, "cluster_count": len(blocks),
            "resource_layer_counts": dict(sorted(counts.items())), "errors": errors}


def collect_result(work, row, net, metadata, stem):
    place = work / f"{stem}.place"
    route = work / f"{stem}.route"
    row["place_audit"] = validate_place(net, place, metadata) if place.is_file() else None
    row["packed_clusters"] = len(ET.parse(net).getroot().findall("block"))
    row["packed_atoms"] = sum(
        1 for block in ET.parse(net).getroot().iter("block")
        if block.attrib.get("name") != "open" and not block.findall("block")
    )
    if row["routed"] and route.is_file():
        ids = re.findall(r"Node:\s*(\d+)\s+CHANZ\b", route.read_text())
        row["routed_unique_chanz_nodes"] = len(set(ids))
    # Partial failed routes must never be treated as completed-route QoR labels.
    if not row["routed"]:
        row["final_cpd_ns"] = None
    row["passed"] = bool(row["vpr_succeeded"] and row["routed"]
                         and row["place_audit"] and row["place_audit"]["passed"])
    save(work / "manifest.json", row)
    return row


def main():
    from heterogeneity_fixtures import CASES, generate_case

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtr-root", required=True, type=Path)
    parser.add_argument("--patched-binary", required=True, type=Path)
    parser.add_argument("--build-manifest", type=Path,
                        help="optional isolated-build provenance; binary hashes must match")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), default=list(CASES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13])
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--original-controls", nargs="*", default=["shifted", "single_site", "no_dsp_layer1"])
    parser.add_argument("--ap-smoke", nargs="*", default=["shifted", "asymmetric"])
    args = parser.parse_args()
    root, binary, out = args.vtr_root.resolve(), args.patched_binary.resolve(), args.out.resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT or out.exists() or (args.archive and args.archive.exists()):
        parser.error("requires pinned VTR commit and fresh output/archive directories")
    out.mkdir(parents=True)
    original = root / "vpr/vpr"
    if args.build_manifest:
        build = json.loads(args.build_manifest.read_text())
        if build["vtr_commit"] != COMMIT or build["probe_binary_sha256"] != sha(binary):
            parser.error("build manifest does not match the pinned patched binary")
        shutil.copy2(args.build_manifest, out / "build_manifest.json")
    tracked = [original, root / "vpr/src/place/move_utils.cpp", root / "vpr/src/place/initial_placement.cpp"]
    before = {str(path): sha(path) for path in tracked}
    report = {"vtr_commit": actual, "patched_binary": str(binary), "patched_binary_sha256": sha(binary),
              "original_binary": str(original), "original_binary_sha256": sha(original),
              "source_arch": SOURCE_ARCH, "source_arch_sha256": sha(root / SOURCE_ARCH),
              "width": args.width, "seeds": args.seeds, "cases": {},
              "scope": "same architecture XML tile definitions; physical distributions differ by layer",
              "purpose": "support and robustness regression, not TailRisk efficacy or timing benchmarking"}
    for name in args.cases:
        work = out / name
        fixture = generate_case(root / SOURCE_ARCH, work, name)
        arch, blif, metadata = fixture["arch"], fixture["blif"], fixture["metadata"]
        base = [str(arch), str(blif), "--device", metadata.get("device_name", "probe"),
                "--route_chan_width", str(args.width)]
        pack_dir = work / "pack"
        pack_dir.mkdir()
        pack_row = run([str(original), *base, "--pack", "--seed", str(args.seeds[0])], pack_dir, args.timeout)
        net = pack_dir / f"{blif.stem}.net"
        case = {"metadata": metadata, "arch_sha256": sha(arch), "blif_sha256": sha(blif),
                "pack": pack_row, "patched": {}, "original_control": None}
        report["cases"][name] = case
        if pack_row["vpr_succeeded"] and net.is_file():
            # Large stress fixture is a support check; no extra seeds are needed.
            seeds = args.seeds[:1] if name == "larger" else args.seeds
            for seed in seeds:
                trial = work / f"patched_seed{seed}"
                trial.mkdir()
                row = run([str(binary), *base, "--net_file", str(net), "--seed", str(seed),
                           "--place", "--route", "--analysis"], trial, args.timeout)
                case["patched"][str(seed)] = collect_result(trial, row, net, metadata, blif.stem)
                save(out / "summary.json", report)
            if name in args.original_controls:
                trial = work / "original_control"
                trial.mkdir()
                row = run([str(original), *base, "--net_file", str(net), "--seed", str(args.seeds[0]),
                           "--place", "--route", "--analysis"], trial, args.timeout)
                case["original_control"] = collect_result(trial, row, net, metadata, blif.stem)
            if name in args.ap_smoke:
                import cluster_sa_probe as sa
                initial = work / f"patched_seed{seeds[0]}" / f"{blif.stem}.place"
                if initial.is_file():
                    trial = work / "ap_flatrecon_roundtrip"
                    trial.mkdir()
                    logic = sa.read_net(net)
                    positions = sa.read_place(initial, logic)
                    fplace = trial / "input.fplace"
                    sa.write_fplace(fplace, logic, positions)
                    if sa.read_fplace(fplace, logic) != positions:
                        raise AssertionError("cluster-to-atom lowering changed input positions")
                    trial_net = trial / f"{blif.stem}.net"
                    shutil.copy2(net, trial_net)
                    row = run([str(binary), *base, "--net_file", str(trial_net), "--seed", str(seeds[0]),
                               "--read_flat_place", str(fplace), "--analytical_place", "--route", "--analysis",
                               "--ap_full_legalizer", "flat-recon", "--ap_detailed_placer", "annealer",
                               "--write_flat_place", str(trial / "post_route.fplace"),
                               "--write_legalized_flat_place", str(trial / "post_legalizer.fplace")],
                              trial, args.timeout)
                    case["ap_roundtrip"] = collect_result(trial, row, trial_net, metadata, blif.stem)
                    case["ap_roundtrip"]["nonzero_layer_input_atoms"] = sum(
                        len(block["atoms"]) for block, point in zip(logic["clusters"], positions) if point[3] != 0)
                    case["ap_roundtrip"]["passed"] &= case["ap_roundtrip"]["nonzero_layer_input_atoms"] > 0
                    save(trial / "manifest.json", case["ap_roundtrip"])
        save(out / "summary.json", report)
    report["original_artifacts_before"] = before
    report["original_artifacts_after"] = {str(path): sha(path) for path in tracked}
    report["original_unchanged"] = before == report["original_artifacts_after"]
    patched_trials = [row for case in report["cases"].values() for row in case["patched"].values()]
    report["patched_trial_count"] = len(patched_trials)
    report["passed_trial_count"] = sum(row["passed"] for row in patched_trials)
    report["passed"] = bool(report["original_unchanged"] and patched_trials
                            and all(case["pack"]["vpr_succeeded"] and case["patched"] for case in report["cases"].values())
                            and all(row["passed"] for row in patched_trials)
                            and all(case.get("ap_roundtrip", {"passed": True})["passed"] for case in report["cases"].values()))
    save(out / "summary.json", report)
    if args.archive:
        shutil.copytree(out, args.archive)
    print(json.dumps({key: report[key] for key in (
        "patched_trial_count", "passed_trial_count", "original_unchanged", "passed")}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
