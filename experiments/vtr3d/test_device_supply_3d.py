"""Independent tiny RRG fixtures for directional/shared 3D capacity semantics."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from device_supply_3d import build_device_supply, region_at


def fixture(folder, *, width=6, height=2, layers=2):
    arch = ET.Element("architecture")
    tiles = ET.SubElement(arch, "tiles")
    for name in ("clb", "memory", "dsp_top", "io"):
        tile = ET.SubElement(tiles, "tile", name=name, width="1", height="1")
        ET.SubElement(tile, "sub_tile", name=name, capacity="2" if name == "io" else "1")
    arch_path = folder / "arch.xml"
    ET.ElementTree(arch).write(arch_path)
    root = ET.Element("rr_graph")
    channels = ET.SubElement(root, "channels")
    ET.SubElement(channels, "channel", chan_width_max="80")
    segments = ET.SubElement(root, "segments")
    ET.SubElement(segments, "segment", id="0", name="L4")
    ET.SubElement(segments, "segment", id="1", name="vertical")
    types = ET.SubElement(root, "block_types")
    for i, name in enumerate(("EMPTY", "clb", "memory", "dsp_top", "io")):
        ET.SubElement(types, "block_type", id=str(i), name=name, width="1", height="1")
    grid = ET.SubElement(root, "grid")
    for z in range(layers):
        for x in range(width):
            for y in range(height):
                kind = 4 if (x, y) == (0, 0) else 3 if (x, y, z) == (1, 1, 0) else 2 if (x, y, z) == (2, 1, layers - 1) else 1
                ET.SubElement(grid, "grid_loc", x=str(x), y=str(y), layer=str(z),
                              block_type_id=str(kind), width_offset="0", height_offset="0")
    nodes = ET.SubElement(root, "rr_nodes")
    edges = ET.SubElement(root, "rr_edges")
    return arch_path, root, nodes, edges


def wire(nodes, node_id, kind, direction, *, x0=0, x1=5, y0=0, y1=0, z0=0, z1=0, capacity=1):
    node = ET.SubElement(nodes, "node", id=str(node_id), type=kind,
                         direction=direction, capacity=str(capacity))
    ET.SubElement(node, "loc", xlow=str(x0), xhigh=str(x1), ylow=str(y0), yhigh=str(y1),
                  layer_low=str(z0), layer_high=str(z1))
    ET.SubElement(node, "segment", segment_id="1" if kind == "CHANZ" else "0")
    return node


def connect(edges, source, sink, repeats=1):
    for _ in range(repeats):
        ET.SubElement(edges, "edge", src_node=str(source), sink_node=str(sink), switch_id="0")


class DeviceSupplyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def build(self, root, arch, **kwargs):
        rrg = self.folder / "rr.xml"
        ET.ElementTree(root).write(rrg)
        return build_device_supply(rrg, arch, columns=3, rows=2, **kwargs)

    def test_shared_long_wire_and_bidirectional_supply_count_once(self):
        arch, root, nodes, edges = fixture(self.folder)
        wire(nodes, 0, "CHANX", "INC_DIR")
        wire(nodes, 1, "CHANX", "INC_DIR")
        wire(nodes, 2, "CHANX", "BI_DIR", capacity=3)
        connect(edges, 0, 1, repeats=40)
        device, supply = self.build(root, arch)
        inc = next(p for p in supply["pools"] if p["direction"] == "INC_DIR")
        bi = next(p for p in supply["pools"] if p["direction"] == "BI_DIR")
        self.assertEqual(inc["arcs"], [[0, 1], [1, 2]])
        self.assertEqual((inc["node_count"], inc["capacity"]), (2, 2))
        self.assertEqual(bi["arcs"], [[0, 1], [1, 0], [1, 2], [2, 1]])
        self.assertEqual(bi["capacity"], 3)  # not 3 times the number of cuts/directions
        self.assertEqual(sum(p["capacity"] for p in supply["pools"]), 5)
        self.assertEqual(supply["rr_edge_count"], 40)
        self.assertEqual(supply["extraction"]["retained_rr_edges"], 0)
        self.assertEqual(supply["channel_width"], 80)
        self.assertEqual(device["grid"], {"width": 6, "height": 2, "layers": 2})

    def test_direction_and_isolated_vertical_nodes(self):
        arch, root, nodes, edges = fixture(self.folder)
        wire(nodes, 0, "CHANX", "INC_DIR")
        wire(nodes, 1, "CHANX", "INC_DIR", z0=1, z1=1)
        wire(nodes, 2, "CHANZ", "INC_DIR", x1=0, z1=1)
        wire(nodes, 3, "CHANZ", "DEC_DIR", x1=0, z1=1)
        wire(nodes, 4, "CHANZ", "BI_DIR", x1=0, z1=1, capacity=5)
        connect(edges, 0, 2)
        connect(edges, 2, 1)
        connect(edges, 1, 3)
        connect(edges, 3, 0)
        connect(edges, 4, 4)  # self-loop does not create terminal connectivity
        device, supply = self.build(root, arch)
        vertical = [p for p in supply["pools"] if p["kind"] == "CHANZ"]
        self.assertEqual(len(vertical), 2)
        self.assertEqual(next(p for p in vertical if p["direction"] == "INC_DIR")["arcs"], [[0, 6]])
        self.assertEqual(next(p for p in vertical if p["direction"] == "DEC_DIR")["arcs"], [[6, 0]])
        audit = supply["vertical_connection_audit"]
        self.assertEqual((audit["candidate_node_count"], audit["included_node_count"], audit["excluded_node_count"]), (3, 2, 1))
        self.assertEqual(audit["included_capacity"], 2)

    def test_actual_slot_capacity_and_absent_layer_resource(self):
        arch, root, nodes, edges = fixture(self.folder)
        wire(nodes, 0, "CHANX", "INC_DIR")
        device, supply = self.build(root, arch)
        self.assertEqual(device["layer_resource_capacity"][0]["dsp_top"], 1)
        self.assertEqual(device["layer_resource_capacity"][1]["dsp_top"], 0)
        self.assertEqual(device["total_resource_capacity"]["io"], 4)
        slots = [s for r in device["regions"] for s in r["tile_coordinates"]["io"]]
        self.assertEqual(sorted(slots), [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]])
        self.assertEqual(region_at(device, 5, 1, 1), 11)
        self.assertEqual(region_at(device, 1, 1, 0), 3)
        with self.assertRaises(ValueError):
            region_at(device, 0, 0, 2)
        with self.assertRaises(ValueError):
            region_at(device, -1, 0, 0)

    def test_zero_capacity_is_absent_not_epsilon(self):
        arch, root, nodes, edges = fixture(self.folder)
        wire(nodes, 0, "CHANX", "INC_DIR", capacity=0)
        device, supply = self.build(root, arch)
        self.assertEqual(supply["pools"], [])

    def test_active_multisite_tile_rejected(self):
        arch, root, nodes, edges = fixture(self.folder)
        tree = ET.parse(arch)
        tree.getroot().find("tiles/tile[@name='clb']").set("height", "2")
        tree.write(arch)
        root.find("block_types/block_type[@name='clb']").set("height", "2")
        wire(nodes, 0, "CHANX", "INC_DIR")
        with self.assertRaisesRegex(ValueError, "multi-site"):
            self.build(root, arch)

    def test_duplicate_grid_and_rr_nodes_rejected(self):
        arch, root, nodes, edges = fixture(self.folder)
        wire(nodes, 0, "CHANX", "INC_DIR")
        wire(nodes, 0, "CHANX", "INC_DIR")
        with self.assertRaisesRegex(ValueError, "unique strictly increasing"):
            self.build(root, arch)
        nodes.remove(nodes[-1])
        grid = root.find("grid")
        grid.append(ET.fromstring(ET.tostring(grid[0])))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.build(root, arch)

    def test_one_layer_mode_and_bounded_footprint_cache(self):
        arch, root, nodes, edges = fixture(self.folder, layers=1)
        wire(nodes, 0, "CHANX", "INC_DIR")
        wire(nodes, 1, "CHANX", "DEC_DIR")
        wire(nodes, 2, "CHANX", "BI_DIR")
        device, supply = self.build(root, arch, max_footprint_cache=1)
        self.assertEqual(device["grid"]["layers"], 1)
        self.assertEqual(supply["layer_count"], 1)
        self.assertEqual(supply["extraction"]["cache_entries"], 1)
        self.assertEqual(len(supply["pools"]), 3)
        json.dumps((device, supply), allow_nan=False)

    def test_unknown_direction_and_nonfinite_capacity_rejected(self):
        arch, root, nodes, edges = fixture(self.folder)
        node = wire(nodes, 0, "CHANX", "NONE")
        with self.assertRaisesRegex(ValueError, "direction"):
            self.build(root, arch)
        node.set("direction", "INC_DIR")
        node.set("capacity", "NaN")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.build(root, arch)

    def test_deadline_and_vertical_memory_limits_fail_explicitly(self):
        arch, root, nodes, edges = fixture(self.folder)
        wire(nodes, 0, "CHANZ", "INC_DIR", x1=0, z1=1)
        wire(nodes, 1, "CHANZ", "INC_DIR", x1=0, z1=1)
        with self.assertRaisesRegex(ValueError, "memory limit"):
            self.build(root, arch, max_vertical_nodes=1)
        with self.assertRaises(TimeoutError):
            self.build(root, arch, max_seconds=1e-12)


if __name__ == "__main__":
    unittest.main()
