#!/usr/bin/env python3
"""Check that the isolated 3D hardening keeps the original 2D flow usable."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from heterogeneity_fixtures import generate_case
from probe_layer_heterogeneity import COMMIT, run, save, sha
from validate_layer_heterogeneity import collect_result
import cluster_sa_probe as sa


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vtr-root", type=Path, required=True)
    p.add_argument("--patched-binary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--archive", type=Path)
    a = p.parse_args()
    root, out, binary = a.vtr_root.resolve(), a.out.resolve(), a.patched_binary.resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT or out.exists() or (a.archive and a.archive.exists()):
        p.error("requires fixed commit and new output/archive directories")
    source = root / "vtr_flow/arch/multi_die/hecate/2d/hecate_2d_k6FracN10LB_mem20K_complexDSP_customSB_7nm.xml"
    fixture = generate_case(source, out, "aligned")
    arch, blif, metadata = fixture["arch"], fixture["blif"], fixture["metadata"]
    tree = ET.parse(arch)
    fixed = tree.getroot().find("layout/fixed_layout")
    fixed.remove(fixed.findall("layer")[1])
    tree.write(arch, encoding="utf-8", xml_declaration=True)
    metadata["layers"] = metadata["layers"][:1]
    metadata["layer_count"] = 1
    metadata["architecture_sha256"] = sha(arch)
    metadata["capacity_total"] = metadata["layers"][0]["capacity"]
    metadata["resource_available_layers"] = {kind: [0] for kind in metadata["capacity_total"]}
    save(out / "fixture.json", metadata)
    original = root / "vpr/vpr"
    before = sha(original)
    base = [str(arch), str(blif), "--device", "probe", "--route_chan_width", "320"]
    pack = out / "pack"
    pack.mkdir()
    packing = run([str(original), *base, "--pack", "--seed", "11"], pack, 90)
    net = pack / "tiny.net"
    report = {"vtr_commit": commit, "original_binary_sha256": before,
              "patched_binary_sha256": sha(binary), "pack": packing, "trials": {}, "pairs": {}}
    if not packing["vpr_succeeded"]:
        save(out / "summary.json", report)
        raise SystemExit(1)
    logic = sa.read_net(net)
    for seed in (11, 12, 13):
        for method, executable in (("original", original), ("patched", binary)):
            folder = out / f"{method}_seed{seed}"
            folder.mkdir()
            row = run([str(executable), *base, "--net_file", str(net), "--seed", str(seed),
                       "--place", "--route", "--analysis"], folder, 90)
            report["trials"][f"{method}_seed{seed}"] = collect_result(folder, row, net, metadata, "tiny")
        first, second = out / f"original_seed{seed}", out / f"patched_seed{seed}"
        comparable = all(report["trials"][f"{method}_seed{seed}"]["passed"] for method in ("original", "patched"))
        report["pairs"][str(seed)] = {
            "comparable": comparable,
            "cluster_positions_identical": (sa.read_place(first / "tiny.place", logic) == sa.read_place(second / "tiny.place", logic)) if comparable else None,
            "note": "Identical coordinates are diagnostic, not required: the inclusive search fix may change RNG trajectories."}
        save(out / "summary.json", report)
    # Exercise the unchanged atom-level input format through the same full
    # legalizer used by the existing 2D training pipeline.
    folder = out / "patched_2d_fplace_roundtrip"
    folder.mkdir()
    fplace = folder / "input.fplace"
    sa.write_fplace(fplace, logic, sa.read_place(out / "patched_seed11/tiny.place", logic))
    trial_net = folder / "tiny.net"
    shutil.copy2(net, trial_net)
    row = run([str(binary), *base, "--net_file", str(trial_net), "--seed", "11",
               "--analytical_place", "--read_flat_place", str(fplace), "--ap_full_legalizer", "flat-recon",
               "--ap_detailed_placer", "annealer", "--route", "--analysis",
               "--write_flat_place", str(folder / "post_route.fplace")], folder, 90)
    report["trials"]["fplace_roundtrip"] = collect_result(folder, row, trial_net, metadata, "tiny")
    report["original_unchanged"] = before == sha(original)
    report["passed"] = all(row["passed"] for row in report["trials"].values()) and report["original_unchanged"]
    save(out / "summary.json", report)
    if a.archive:
        shutil.copytree(out, a.archive)
    print(json.dumps({"passed": report["passed"], "trial_count": len(report["trials"]),
                      "pairs": report["pairs"], "original_unchanged": report["original_unchanged"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
