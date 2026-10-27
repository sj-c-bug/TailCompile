#!/usr/bin/env python3
"""Versioned 3D tile slots and shared directional wire pools from a real RRG.

This is a coarse supply proxy, not a detailed routability certificate. Wire
capacity is counted once per RR node, even when its span crosses several cuts
or supports both directions. RR switch-edge multiplicity never adds capacity.
Pin access, switch-only region crossings and exact track-to-track reachability
remain outside the proxy. The original 2D formats and checkpoints are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

VTR_COMMIT = "40f8d19708b0b60e2aa04452deb1cfed68d96b0b"
RESOURCES = ("clb", "memory", "dsp_top", "io")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def region_at(device, x, y, layer):
    """O(1) whole-grid bins, including perimeter IO; no hidden layer clamping."""
    grid, bins = device["grid"], device["region_grid"]
    if (not all(math.isfinite(float(v)) for v in (x, y, layer))
            or int(layer) != layer or not 0 <= layer < grid["layers"]
            or not 0 <= x < grid["width"] or not 0 <= y < grid["height"]):
        raise ValueError(f"coordinate outside device: {(x, y, layer)}")
    column = min(bins["columns"] - 1, int(x * bins["columns"] / grid["width"]))
    row = min(bins["rows"] - 1, int(y * bins["rows"] / grid["height"]))
    return int(layer) * bins["rows"] * bins["columns"] + row * bins["columns"] + column


def _architecture_tiles(path):
    root = ET.parse(path).getroot()
    tiles = {}
    for tile in root.findall("tiles/tile"):
        name = tile.attrib["name"]
        subtiles = tile.findall("sub_tile")
        if not subtiles:
            raise ValueError(f"architecture tile {name} has no sub_tile definition")
        capacity = sum(int(sub.get("capacity", "1")) for sub in subtiles)
        if capacity <= 0:
            raise ValueError(f"architecture tile {name} has nonpositive capacity")
        tiles[name] = {"width": int(tile.get("width", "1")),
                       "height": int(tile.get("height", "1")), "capacity": capacity}
    if not tiles:
        raise ValueError("requires the pinned VTR architecture tiles/sub_tile format")
    return tiles


def _make_device(grid_rows, block_types, arch_tiles, columns, rows):
    if not grid_rows:
        raise ValueError("RRG contains no grid locations before rr_nodes")
    width = max(p[0] for p in grid_rows) + 1
    height = max(p[1] for p in grid_rows) + 1
    layers = max(p[2] for p in grid_rows) + 1
    if not 1 <= columns <= width or not 1 <= rows <= height:
        raise ValueError("region columns/rows must be positive and no larger than the grid")
    seen = {(x, y, z) for x, y, z, *_ in grid_rows}
    if len(seen) != len(grid_rows) or len(seen) != width * height * layers:
        raise ValueError("RRG grid must contain exactly one record for every XYZ location")
    device = {"schema": "tailcompile-device-ir-3d-v2",
              "grid": {"width": width, "height": height, "layers": layers},
              "region_grid": {"columns": columns, "rows": rows, "layers": layers},
              "resources": list(RESOURCES), "slot_order": ["x", "y", "subtile", "layer"],
              "regions": [], "capacity_semantics": "actual base-site anchors and XML sub_tile capacities; no proportional fill quotas"}
    for z in range(layers):
        for row in range(rows):
            for col in range(columns):
                xl = math.ceil(col * width / columns)
                xh = math.ceil((col + 1) * width / columns) - 1
                yl = math.ceil(row * height / rows)
                yh = math.ceil((row + 1) * height / rows) - 1
                device["regions"].append({
                    "id": len(device["regions"]), "column": col, "row": row, "layer": z,
                    "bounds": {"x_low": xl, "x_high": xh, "y_low": yl, "y_high": yh},
                    "center": [(xl + xh) / 2, (yl + yh) / 2, z],
                    "tile_coordinates": {kind: [] for kind in RESOURCES},
                    "tile_counts": {kind: 0 for kind in RESOURCES}})
    actual_types = Counter()
    for x, y, z, type_id, dx, dy in grid_rows:
        if type_id not in block_types:
            raise ValueError(f"unknown RRG block_type_id {type_id}")
        tile = block_types[type_id]
        name = tile["name"]
        if name.upper() == "EMPTY":
            continue
        if name not in RESOURCES or name not in arch_tiles:
            raise ValueError(f"active tile {name!r} is outside the supported resource adapter")
        spec = arch_tiles[name]
        if (tile["width"], tile["height"]) != (spec["width"], spec["height"]):
            raise ValueError(f"RRG/XML tile dimensions disagree for {name}")
        if spec["width"] != 1 or spec["height"] != 1:
            raise ValueError(f"active multi-site tile {name} requires macro/base-site support")
        if dx or dy:
            raise ValueError(f"unit tile {name} has a nonzero RRG base-site offset")
        region = device["regions"][region_at(device, x, y, z)]
        slots = [[x, y, sub, z] for sub in range(spec["capacity"])]
        region["tile_coordinates"][name].extend(slots)
        region["tile_counts"][name] += len(slots)
        actual_types[name] += len(slots)
    device["total_resource_capacity"] = {kind: actual_types[kind] for kind in RESOURCES}
    device["layer_resource_capacity"] = [
        {kind: sum(r["tile_counts"][kind] for r in device["regions"] if r["layer"] == z)
         for kind in RESOURCES} for z in range(layers)]
    return device


def _footprint(device, kind, loc, direction):
    xl, xh, yl, yh, zl, zh = loc
    if xl > xh or yl > yh or zl > zh:
        raise ValueError("RRG channel has reversed geometric bounds")
    if kind in ("CHANX", "CHANY") and zl != zh:
        raise ValueError(f"{kind} unexpectedly spans multiple layers")
    if kind == "CHANX" and yl != yh or kind == "CHANY" and xl != xh:
        raise ValueError("in-plane wire is not axis-aligned")
    if kind == "CHANZ" and (xl != xh or yl != yh):
        raise ValueError("vertical wire with XY displacement needs an explicit adapter")
    if kind == "CHANX":
        points = ((x, yl, zl) for x in range(xl, xh + 1))
    elif kind == "CHANY":
        points = ((xl, y, zl) for y in range(yl, yh + 1))
    else:
        points = ((xl, yl, z) for z in range(zl, zh + 1))
    sequence = []
    for point in points:
        rid = region_at(device, *point)
        if not sequence or sequence[-1] != rid:
            sequence.append(rid)
    arcs = list(zip(sequence, sequence[1:]))
    if direction == "DEC_DIR":
        arcs = [(b, a) for a, b in arcs]
    elif direction == "BI_DIR":
        arcs += [(b, a) for a, b in arcs]
    elif direction != "INC_DIR":
        raise ValueError(f"unsupported channel direction {direction!r}")
    return tuple(sorted(arcs))


def build_device_supply(rr_graph, architecture, columns=4, rows=4, *,
                        max_seconds=300.0, max_footprint_cache=100000,
                        max_vertical_nodes=2000000):
    """Read actual RRG slots and pools with streaming edge disposal.

    VPR writes metadata/grid before rr_nodes. That ordering is checked, rather
    than guessed from the XML filename. Only CHANZ connection flags are retained
    while scanning edges; all rr_edges are immediately discarded. Vertical nodes
    lacking either a non-self input or output are excluded. Nonzero degree is
    necessary, not proof of detailed source-to-sink reachability.

    The deadline covers extraction and final hashing; it raises TimeoutError,
    never returns a partial device. Cache and vertical-node limits fail clearly
    before unbounded growth. These are preparation limits, not search budgets.
    """
    rr_graph, architecture = Path(rr_graph), Path(architecture)
    if max_seconds <= 0 or max_footprint_cache <= 0 or max_vertical_nodes <= 0:
        raise ValueError("preparation limits must be positive")
    started = time.monotonic()
    deadline = started + max_seconds
    arch_tiles = _architecture_tiles(architecture)
    grid_rows, block_types, segments = [], {}, {}
    device, channel_width = None, None
    pools = {}
    footprints = OrderedDict()
    vertical = {}
    node_count, edge_count, event_count, previous_node_id = 0, 0, 0, -1
    counts = Counter()
    cache_hits = cache_misses = 0

    def add_wire(key, capacity, node_id):
        pool = pools.setdefault(key, {"capacity": 0.0, "node_count": 0,
                                      "example_rr_node_ids": [], "rr_node_id_min": node_id,
                                      "rr_node_id_max": node_id})
        pool["capacity"] += capacity
        pool["node_count"] += 1
        pool["rr_node_id_min"] = min(pool["rr_node_id_min"], node_id)
        pool["rr_node_id_max"] = max(pool["rr_node_id_max"], node_id)
        if len(pool["example_rr_node_ids"]) < 8:
            pool["example_rr_node_ids"].append(node_id)

    opener = gzip.open if rr_graph.suffix == ".gz" else open
    with opener(rr_graph, "rb") as stream:
        stack = []
        for event, item in ET.iterparse(stream, events=("start", "end")):
            event_count += 1
            if event_count % 16384 == 0 and time.monotonic() >= deadline:
                raise TimeoutError(f"3D supply preparation exceeded {max_seconds}s")
            if event == "start":
                stack.append(item)
                if item.tag == "rr_nodes":
                    device = _make_device(grid_rows, block_types, arch_tiles, int(columns), int(rows))
                    grid_rows.clear()
                continue
            parent = stack[-2] if len(stack) > 1 else None
            parent_tag = parent.tag if parent is not None else None
            if item.tag == "channel" and parent_tag == "channels":
                channel_width = int(item.attrib["chan_width_max"])
            elif item.tag == "block_type" and parent_tag == "block_types":
                block_types[int(item.attrib["id"])] = {
                    "name": item.attrib["name"], "width": int(item.get("width", "1")),
                    "height": int(item.get("height", "1"))}
            elif item.tag == "grid_loc" and parent_tag == "grid":
                grid_rows.append(tuple(int(item.get(key, "0")) for key in
                                       ("x", "y", "layer", "block_type_id", "width_offset", "height_offset")))
            elif item.tag == "segment" and parent_tag == "segments":
                segments[item.attrib["id"]] = item.get("name", item.attrib["id"])
            elif item.tag == "node" and parent_tag == "rr_nodes":
                if device is None:
                    raise ValueError("RRG grid metadata must precede rr_nodes")
                node_count += 1
                node_id, kind = int(item.attrib["id"]), item.get("type")
                if node_id <= previous_node_id:
                    raise ValueError("requires unique strictly increasing VPR RR node IDs")
                previous_node_id = node_id
                counts[kind] += 1
                if kind in ("CHANX", "CHANY", "CHANZ"):
                    capacity = float(item.get("capacity", "1"))
                    if not math.isfinite(capacity) or capacity < 0:
                        raise ValueError("RRG wire capacity must be finite and nonnegative")
                    if capacity > 0:
                        loc = item.find("loc")
                        segment = item.find("segment")
                        if loc is None or segment is None:
                            raise ValueError(f"wire {node_id} missing location or segment")
                        ll = int(loc.get("layer_low", loc.get("layer", "0")))
                        lh = int(loc.get("layer_high", loc.get("layer", "0")))
                        coords = tuple(int(loc.attrib[k]) for k in ("xlow", "xhigh", "ylow", "yhigh")) + (ll, lh)
                        direction = item.get("direction")
                        geometry = (kind, coords, direction)
                        if geometry in footprints:
                            arcs = footprints[geometry]
                            footprints.move_to_end(geometry)
                            cache_hits += 1
                        else:
                            arcs = _footprint(device, kind, coords, direction)
                            footprints[geometry] = arcs
                            cache_misses += 1
                            if len(footprints) > max_footprint_cache:
                                footprints.popitem(last=False)
                        if arcs:
                            sid = segment.attrib["segment_id"]
                            key = (kind, direction, segments.get(sid, sid), arcs)
                            if kind == "CHANZ":
                                if len(vertical) >= max_vertical_nodes:
                                    raise ValueError("CHANZ connection-audit memory limit exceeded")
                                vertical[node_id] = [key, capacity, False, False]
                            else:
                                add_wire(key, capacity, node_id)
            elif item.tag == "edge" and parent_tag == "rr_edges":
                edge_count += 1
                source, sink = int(item.attrib["src_node"]), int(item.attrib["sink_node"])
                if source != sink:
                    if source in vertical:
                        vertical[source][3] = True
                    if sink in vertical:
                        vertical[sink][2] = True
            # Keep node-local loc/segment and block_type pin definitions until
            # their enclosing record closes, then unlink to bound XML memory.
            if parent is not None and parent_tag in (
                    "rr_nodes", "rr_edges", "grid", "segments", "channels", "block_types", "switches"):
                parent.remove(item)
            stack.pop()
    if device is None or channel_width is None or channel_width <= 0:
        raise ValueError("RRG missing device grid, nodes, or positive channel width")
    disconnected = []
    vertical_connected_capacity = 0.0
    for node_id, (key, capacity, incoming, outgoing) in vertical.items():
        if incoming and outgoing:
            add_wire(key, capacity, node_id)
            vertical_connected_capacity += capacity
        elif len(disconnected) < 16:
            disconnected.append({"rr_node_id": node_id, "has_input": incoming, "has_output": outgoing})
    rr_hash, arch_hash = digest(rr_graph), digest(architecture)
    if time.monotonic() >= deadline:
        raise TimeoutError(f"3D supply preparation exceeded {max_seconds}s")
    provenance = {"vtr_commit": VTR_COMMIT, "vtr_commit_source": "caller must independently verify executable/build manifest",
                  "rr_graph": str(rr_graph.resolve()), "rr_graph_sha256": rr_hash,
                  "architecture": str(architecture.resolve()), "architecture_sha256": arch_hash}
    device["source"] = provenance.copy()
    device["limitations"] = ["active multi-site tiles and coupled placement macros are rejected, not modeled",
                              "tile capacity does not establish detailed pin or track reachability"]
    supply = {
        "schema": "tailcompile-3d-wire-pools-v2", "provenance": provenance,
        "channel_width": channel_width, "layer_count": device["grid"]["layers"],
        "region_count": len(device["regions"]), "rr_node_count": node_count, "rr_edge_count": edge_count,
        "rr_node_counts": dict(sorted(counts.items())),
        "device_ir_content_sha256": hashlib.sha256(json.dumps(device, sort_keys=True).encode()).hexdigest(),
        "capacity_semantics": "unique RR wire-node capacity shared over the entire directed boundary footprint; BI_DIR shares one pool",
        "vertical_connection_audit": {
            "candidate_node_count": len(vertical),
            "included_node_count": sum(v[2] and v[3] for v in vertical.values()),
            "included_capacity": vertical_connected_capacity,
            "excluded_node_count": sum(not (v[2] and v[3]) for v in vertical.values()),
            "excluded_sample": disconnected,
            "criterion": "non-self incoming AND outgoing RR edge; necessary but not sufficient for exact routability"},
        "limits": {"max_seconds": max_seconds, "max_footprint_cache": max_footprint_cache,
                   "max_vertical_nodes": max_vertical_nodes},
        "extraction": {"elapsed_seconds": time.monotonic() - started, "cache_hits": cache_hits,
                       "cache_misses": cache_misses, "cache_entries": len(footprints),
                       "retained_rr_edges": 0},
        "limitations": ["coarse region reachability is not exact RRG reachability",
                        "pin access and switch-only region crossings are not modeled",
                        "fractional pool sharing is a proxy, not a legal detailed route",
                        "nonzero CHANZ degree alone does not prove terminal reachability",
                        "per-width actual RRG required; capacity is not linearly extrapolated"],
        "pools": [{"id": i, "kind": key[0], "direction": key[1], "segment": key[2],
                   "arcs": [list(arc) for arc in key[3]], **value}
                  for i, (key, value) in enumerate(sorted(pools.items()))]}
    return device, supply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rr-graph", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--device-out", type=Path, required=True)
    parser.add_argument("--supply-out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()
    if args.device_out.exists() or args.supply_out.exists():
        parser.error("refusing to overwrite existing device/supply artifacts")
    device, supply = build_device_supply(args.rr_graph, args.architecture, args.columns, args.rows,
                                         max_seconds=args.max_seconds)
    for path, value in ((args.device_out, device), (args.supply_out, supply)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"regions": len(device["regions"]), "layers": supply["layer_count"],
                      "pools": len(supply["pools"]), "extraction": supply["extraction"]}, indent=2))


if __name__ == "__main__":
    main()
