#!/usr/bin/env python3
"""Small legal cluster annealer for the 10x10 layer-heterogeneity fixture.

This is an experimental baseline, not a general FPGA legalizer: it only
accepts the fixture's single-site, single-subtile logic resources, and keeps
IO fixed. It freezes packing and changes whole-cluster locations.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import xml.etree.ElementTree as ET


def read_net(path):
    root = ET.parse(path).getroot()
    clusters, atom_owner, drivers, sinks = [], {}, {}, defaultdict(set)
    clock_nets = set((root.findtext("clocks") or "").split())
    for block in root.findall("block"):
        if block.get("name") == "open":
            continue
        index = len(clusters)
        atoms = []
        for node in block.iter("block"):
            if node.get("name") == "open" or node.findall("block"):
                continue
            name = node.get("name")
            if name in atom_owner:
                raise ValueError(f"duplicate atom: {name}")
            atoms.append(name)
            atom_owner[name] = index
            for port in node.findall("outputs/port"):
                for net in (port.text or "").split():
                    if net != "open" and "->" not in net:
                        if net in drivers and drivers[net] != index:
                            raise ValueError(f"multiple drivers: {net}")
                        drivers[net] = index
        inputs = set()
        for port in block.findall("inputs/port"):
            inputs.update(net for net in (port.text or "").split() if net != "open")
        for net in inputs - clock_nets:
            sinks[net].add(index)
        clusters.append({"name": block.get("name"), "kind": block.get("instance").split("[")[0], "atoms": atoms})
    edges = []
    for net in sorted(sinks):
        if net not in drivers:
            raise ValueError(f"missing net driver: {net}")
        pins = sorted(sinks[net] | {drivers[net]})
        if len(pins) > 1:
            edges.append({"name": net, "pins": pins})
    incident = [set() for _ in clusters]
    for edge_id, edge in enumerate(edges):
        for pin in edge["pins"]:
            incident[pin].add(edge_id)
    return {"clusters": clusters, "atom_owner": atom_owner, "edges": edges, "incident": incident,
            "clock_nets_excluded": sorted(clock_nets)}


def fixture_slots(arch):
    root = ET.parse(arch).getroot()
    layout = root.find("layout/fixed_layout")
    if layout is None or layout.get("width") != "10" or layout.get("height") != "10":
        raise ValueError("probe only supports the 10x10 fixture")
    capacities = {}
    for tile in root.findall("tiles/tile"):
        if tile.get("name") in ("clb", "memory", "dsp_top", "io") and (int(tile.get("width", "1")) != 1 or int(tile.get("height", "1")) != 1):
            raise ValueError("probe rejects multi-site tiles/macros")
        capacities[tile.get("name")] = sum(int(sub.get("capacity", "1")) for sub in tile.findall("sub_tile"))
    slots = defaultdict(list)
    for layer in layout.findall("layer"):
        z = int(layer.get("die"))
        columns = {int(col.get("startx")): col.get("type") for col in layer.findall("col")}
        if set(columns.values()) != {"dsp_top", "memory"}:
            raise ValueError("fixture must have one DSP and one memory column per layer")
        for x in range(10):
            for y in range(10):
                if x in (0, 9) and y in (0, 9):
                    continue
                kind = "io" if x in (0, 9) or y in (0, 9) else columns.get(x, "clb")
                for sub in range(capacities[kind]):
                    slots[kind].append((x, y, sub, z))
    return dict(slots)


def read_place(path, logic):
    positions = {}
    names = {row["name"] for row in logic["clusters"]}
    for line in Path(path).read_text().splitlines():
        parts = line.split("#", 1)[0].split()
        if parts and parts[0] in names:
            if len(parts) != 5 or parts[0] in positions:
                raise ValueError(f"invalid/duplicate place row: {line}")
            positions[parts[0]] = tuple(map(int, parts[1:]))
    if set(positions) != names:
        raise ValueError("placement cluster set mismatch")
    return [positions[row["name"]] for row in logic["clusters"]]


def read_fplace(path, logic):
    owner = logic["atom_owner"]
    by_cluster, seen = {}, set()
    for line in Path(path).read_text().splitlines():
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        atom = parts[0]
        if len(parts) != 5 or atom not in owner or atom in seen:
            raise ValueError(f"invalid/unknown/duplicate fplace atom: {line}")
        seen.add(atom)
        x, y, layer, sub = map(float, parts[1:])
        if any(value != int(value) for value in (x, y, layer, sub)):
            raise ValueError("probe requires already legalized integer hints")
        point = (int(x), int(y), int(sub), int(layer))
        node = owner[atom]
        if node in by_cluster and by_cluster[node] != point:
            raise ValueError("atoms in a frozen cluster have inconsistent coordinates")
        by_cluster[node] = point
    if seen != set(owner):
        raise ValueError("fplace atom set mismatch")
    return [by_cluster[i] for i in range(len(logic["clusters"]))]


def write_fplace(path, logic, positions):
    lines = ["# Probe legal cluster initial positions expanded to atom-level fplace"]
    for row, (x, y, sub, layer) in zip(logic["clusters"], positions):
        lines.extend(f"{atom} {x} {y} {layer} {sub}" for atom in row["atoms"])
    Path(path).write_text("\n".join(lines) + "\n")


def validate(logic, positions, slots, frozen=None):
    if len(positions) != len(logic["clusters"]):
        raise ValueError("cluster count mismatch")
    allowed = {kind: set(values) for kind, values in slots.items()}
    occupied = set()
    for i, (cluster, point) in enumerate(zip(logic["clusters"], positions)):
        if point not in allowed.get(cluster["kind"], set()):
            raise ValueError(f"incompatible slot for {cluster['name']}: {point}")
        if point in occupied:
            raise ValueError(f"slot collision: {point}")
        occupied.add(point)
        if frozen and i in frozen and point != frozen[i]:
            raise ValueError("fixed IO moved")
    return {"valid": True, "clusters": len(positions),
            "resource_counts": dict(Counter(row["kind"] for row in logic["clusters"])),
            "used_layers": sorted({point[3] for point in positions})}


def write_place(path, net, logic, positions, only_io=False):
    sha = hashlib.sha256(Path(net).read_bytes()).hexdigest()
    lines = [f"Netlist_File: {Path(net).resolve()} Netlist_ID: SHA256:{sha}",
             "Array size: 10 x 10 logic blocks", "", "#block name x y subtile layer"]
    if only_io:
        # --fix_clusters reads bare placement constraint rows, not the full
        # .place file's netlist/grid header.
        lines = ["# Fixed IO cluster locations"]
    for i, (row, point) in enumerate(zip(logic["clusters"], positions)):
        if only_io and row["kind"] != "io":
            continue
        lines.append(row["name"] + " " + " ".join(map(str, point)) + f" #{i}")
    Path(path).write_text("\n".join(lines) + "\n")


def edge_cost(edge, positions, z_weight=2.0):
    pts = [positions[i] for i in edge["pins"]]
    return ((max(p[0] for p in pts) - min(p[0] for p in pts))
            + (max(p[1] for p in pts) - min(p[1] for p in pts))
            + z_weight * (max(p[3] for p in pts) - min(p[3] for p in pts)))


def total_cost(logic, positions):
    return sum(edge_cost(edge, positions) for edge in logic["edges"])


def anneal(logic, initial, slots, seed, steps=12000):
    start = time.monotonic()
    validate(logic, initial, slots)
    rng = random.Random(seed)
    positions = list(initial)
    fixed = {i: positions[i] for i, row in enumerate(logic["clusters"]) if row["kind"] == "io"}
    movable = [i for i in range(len(positions)) if i not in fixed]
    occupancy = {point: i for i, point in enumerate(positions)}
    edge_values = [edge_cost(edge, positions) for edge in logic["edges"]]
    score = best_score = sum(edge_values)
    best = list(positions)
    audit = {"seed": seed, "steps": steps, "objective": "sum of net HPWL_xy + 2 * net layer span; excludes clock nets",
             "initial_cost": score, "accepted": 0, "accepted_cross_layer": 0, "rejected_incompatible": 0,
             "movable_clusters": len(movable), "fixed_io_clusters": len(fixed), "trace": []}
    for step in range(steps):
        node = rng.choice(movable)
        target = rng.choice(slots[logic["clusters"][node]["kind"]])
        old = positions[node]
        if target == old:
            continue
        other = occupancy.get(target)
        if other is not None and (other in fixed or old not in slots[logic["clusters"][other]["kind"]]):
            audit["rejected_incompatible"] += 1
            continue
        affected = logic["incident"][node] | (logic["incident"][other] if other is not None else set())
        previous_values = sum(edge_values[i] for i in affected)
        positions[node] = target
        if other is not None:
            positions[other] = old
        updated = {i: edge_cost(logic["edges"][i], positions) for i in affected}
        delta = sum(updated.values()) - previous_values
        temperature = 20.0 * (0.02 / 20.0) ** (step / max(steps - 1, 1))
        accept = delta <= 0 or rng.random() < math.exp(-delta / temperature)
        if accept:
            score += delta
            edge_values_update = updated.items()
            for i, value in edge_values_update:
                edge_values[i] = value
            occupancy.pop(old)
            occupancy[target] = node
            if other is not None:
                occupancy[old] = other
            audit["accepted"] += 1
            audit["accepted_cross_layer"] += int(old[3] != target[3])
            if score < best_score:
                best_score, best = score, list(positions)
        else:
            positions[node] = old
            if other is not None:
                positions[other] = target
        if step % 2000 == 0:
            if total_cost(logic, positions) != score:
                raise AssertionError("incremental objective drift")
            validate(logic, positions, slots, fixed)
            audit["trace"].append({"step": step, "current_cost": score, "best_cost": best_score})
    audit["final_cost"] = best_score
    audit["full_recomputed_cost"] = total_cost(logic, best)
    audit["legality"] = validate(logic, best, slots, fixed)
    audit["elapsed_seconds"] = round(time.monotonic() - start, 4)
    audit["changed_clusters"] = sum(a != b for a, b in zip(initial, best))
    if audit["full_recomputed_cost"] != best_score:
        raise AssertionError("final objective mismatch")
    return best, audit
