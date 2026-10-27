#!/usr/bin/env python3
"""Independent, VPR-free tests for the heterogeneity fixtures and place audit.

Run ``python -m unittest discover -s experiments/vtr3d -p test_layer_heterogeneity.py -v``.
To use another official source archive, set TAILCOMPILE_HETEROGENEITY_TEST_ARCH
to its XML path. Without it, existing repository archives are used when present;
only the generator tests are skipped if neither archive is available.

The runner imports a Linux-only process helper. On Windows a deliberately
nonfunctional resource-module stub allows testing its pure validate_place()
function; the stub raises if a test accidentally tries to start a VPR run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from heterogeneity_fixtures import CASES, generate_case


def _load_place_validator():
    """Load actual runner source, not a reimplementation of its audit logic."""
    path = HERE / "validate_layer_heterogeneity.py"
    spec = importlib.util.spec_from_file_location("_heterogeneity_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    if importlib.util.find_spec("resource") is None:
        resource_stub = types.ModuleType("resource")
        resource_stub.RLIMIT_CORE = 0

        def no_process_helpers(*args, **kwargs):
            raise AssertionError("VPR process helpers are outside this unit test")

        resource_stub.setrlimit = no_process_helpers
        with mock.patch.dict(sys.modules, {"resource": resource_stub}):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module.validate_place


validate_place = _load_place_validator()


def _signature(node):
    return (node.tag, tuple(sorted(node.attrib.items())), (node.text or "").strip(),
            tuple(_signature(child) for child in node))


def _nonlayout_signature(root):
    return tuple(_signature(child) for child in root if child.tag not in ("layout", "fixed_layout"))


class PlaceAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tailcompile-place-audit-", dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.net, self.place = self.work / "tiny.net", self.work / "tiny.place"
        root = ET.Element("block", name="tiny", instance="FPGA_packed_netlist[0]")
        for name, kind in [("clb0", "clb"), ("clb1", "clb"), ("dsp0", "dsp_top"),
                           ("ram0", "memory"), ("in:clk", "io"), ("open", "clb")]:
            ET.SubElement(root, "block", name=name, instance=f"{kind}[0]")
        ET.ElementTree(root).write(self.net, encoding="utf-8", xml_declaration=True)
        self.metadata = {"layers": [
            {"layer": 0, "slots": {"clb": [[1, 1, 0, 0], [1, 2, 0, 0]],
                                    "dsp_top": [[2, 1, 0, 0]], "memory": [],
                                    "io": [[0, 1, 0, 0], [0, 1, 1, 0]]}},
            {"layer": 1, "slots": {"clb": [[3, 3, 0, 1]], "dsp_top": [],
                                    "memory": [[4, 4, 0, 1]], "io": [[5, 2, 0, 1]]}},
        ]}
        self.positions = {"clb0": (1, 1, 0, 0), "clb1": (3, 3, 0, 1),
                          "dsp0": (2, 1, 0, 0), "ram0": (4, 4, 0, 1),
                          "in:clk": (0, 1, 1, 0)}

    def audit(self, positions=None, extra_lines=()):
        if positions is None:
            positions = self.positions
        lines = ["Netlist_File: tiny.net Netlist_ID: SHA256:unit-test", "Array size: 6 x 6 logic blocks",
                 "", "#block name x y subblk layer block number"]
        lines += [name + "\t" + "\t".join(map(str, pos)) + f"\t#{i}"
                  for i, (name, pos) in enumerate(positions.items())]
        self.place.write_text("\n".join([*lines, *extra_lines]) + "\n", encoding="utf-8")
        return validate_place(self.net, self.place, self.metadata)

    def assert_rejected(self, result, error_fragment):
        self.assertFalse(result["passed"])
        self.assertTrue(any(error_fragment in message for message in result["errors"]), result)

    def test_valid_two_layer_resource_placement(self):
        result = self.audit()
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["cluster_count"], 5)  # The packed-netlist open placeholder is not a cluster.
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["resource_layer_counts"], {
            "clb@0": 1, "clb@1": 1, "dsp_top@0": 1, "io@0": 1, "memory@1": 1})

    def test_collision_between_compatible_clusters(self):
        self.positions["clb1"] = self.positions["clb0"]
        self.assert_rejected(self.audit(), "collision")

    def test_missing_cluster(self):
        del self.positions["ram0"]
        self.assert_rejected(self.audit(), "missing 1 clusters: ['ram0']")

    def test_layer_with_no_compatible_resource(self):
        self.positions["dsp0"] = (2, 1, 0, 1)
        self.assert_rejected(self.audit(), "incompatible or off-grid dsp0")

    def test_nonexistent_layer(self):
        self.positions["ram0"] = (4, 4, 0, 7)
        self.assert_rejected(self.audit(), "incompatible or off-grid ram0")

    def test_resource_type_mismatch(self):
        self.positions["clb0"] = self.positions["dsp0"]
        self.assert_rejected(self.audit(), "incompatible or off-grid clb0")

    def test_outside_grid(self):
        self.positions["clb0"] = (99, 1, 0, 0)
        self.assert_rejected(self.audit(), "incompatible or off-grid clb0")

    def test_invalid_subtile(self):
        self.positions["in:clk"] = (0, 1, 8, 0)
        self.assert_rejected(self.audit(), "incompatible or off-grid in:clk")

    def test_duplicate_cluster_line(self):
        self.assert_rejected(self.audit(extra_lines=["clb0 1 2 0 0 #9"]), "duplicate cluster clb0")

    def test_malformed_coordinate_row_returns_failed_audit(self):
        del self.positions["clb0"]
        for line in ("clb0 1 1 0", "clb0 1 1 0 0 99", "clb0"):
            with self.subTest(line=line):
                self.assert_rejected(self.audit(extra_lines=[line]), "malformed coordinate row for clb0")

    def test_noninteger_coordinate_returns_failed_audit(self):
        self.positions["clb0"] = ("1.5", 1, 0, 0)
        self.assert_rejected(self.audit(), "noninteger coordinate for clb0")

    def test_unknown_coordinate_cluster_is_rejected(self):
        self.assert_rejected(self.audit(extra_lines=["unknown_cluster 1 2 0 0 #8"]),
                             "unknown cluster unknown_cluster")


class FixtureStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("TAILCOMPILE_HETEROGENEITY_TEST_ARCH")
        if configured:
            cls.source = Path(configured)
            if not cls.source.is_file():
                raise FileNotFoundError(f"TAILCOMPILE_HETEROGENEITY_TEST_ARCH not found: {cls.source}")
        else:
            candidates = [HERE / "runs/heterogeneity_probe_20261022_w320/official_source_arch.xml",
                          HERE / "runs/fixture_generator_selftest_20261022/aligned/arch.xml"]
            cls.source = next((path for path in candidates if path.is_file()), None)
            if cls.source is None:
                raise unittest.SkipTest("set TAILCOMPILE_HETEROGENEITY_TEST_ARCH to an official Hecate3D XML")
        cls.source_hash = hashlib.sha256(cls.source.read_bytes()).hexdigest()
        cls.source_signature = _nonlayout_signature(ET.parse(cls.source).getroot())

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="tailcompile-heterogeneity-fixture-", dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)

    def test_case_resource_capacities_and_layout_integrity(self):
        expected = {
            "aligned": [(8, 8), (8, 8)], "shifted": [(8, 8), (8, 8)],
            "single_site": [(1, 1), (1, 1)], "no_dsp_layer1": [(8, 8), (0, 8)],
            "no_bram_layer0": [(8, 0), (8, 8)], "asymmetric": [(16, 1), (1, 16)],
            "sparse": [(4, 4), (4, 4)], "larger": [(152, 38), (38, 152)],
        }
        self.assertEqual(set(CASES), set(expected))
        for name in CASES:
            with self.subTest(case=name):
                result = generate_case(self.source, self.work / name, name)
                self.assertIsInstance(result["arch"], Path)
                self.assertIsInstance(result["blif"], Path)
                metadata = result["metadata"]
                self.assertEqual(json.loads(result["metadata_path"].read_text()), metadata)
                self.assertEqual(metadata["layer_count"], 2)
                root = ET.parse(result["arch"]).getroot()
                self.assertEqual(_nonlayout_signature(root), self.source_signature)
                self.assertEqual(len(root.findall("./layout/fixed_layout")), 1)
                self.assertEqual(len(root.findall("./layout/fixed_layout/layer")), 2)
                self.assertEqual([(row["capacity"]["dsp_top"], row["capacity"]["memory"])
                                  for row in metadata["layers"]], expected[name])
                occupied = set()
                for layer in metadata["layers"]:
                    die = root.find(f"./layout/fixed_layout/layer[@die='{layer['layer']}']")
                    singles = {(node.attrib["type"], int(node.attrib["x"]), int(node.attrib["y"]))
                               for node in die.findall("single")}
                    for kind, slots in layer["slots"].items():
                        self.assertEqual(len(slots), layer["capacity"][kind])
                        for x, y, subtile, layer_index in slots:
                            point = (x, y, subtile, layer_index)
                            self.assertNotIn(point, occupied)
                            occupied.add(point)
                            self.assertEqual(layer_index, layer["layer"])
                            self.assertTrue(0 <= x < metadata["width"])
                            self.assertTrue(0 <= y < metadata["height"])
                            self.assertTrue(0 <= subtile < metadata["tile_subtile_capacities"][kind])
                            if kind in ("dsp_top", "memory"):
                                self.assertIn((kind, x, y), singles)
                    self.assertEqual(len(singles), layer["site_count"]["dsp_top"] + layer["site_count"]["memory"])
                self.assertEqual(sum(metadata["capacity_total"].values()), len(occupied))
                self.assertEqual(metadata["source_arch_sha256"], self.source_hash)
                self.assertEqual(metadata["architecture_sha256"], hashlib.sha256(result["arch"].read_bytes()).hexdigest())
                self.assertEqual(metadata["blif_sha256"], hashlib.sha256(result["blif"].read_bytes()).hexdigest())
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), self.source_hash)

    def test_blif_atom_counts_and_full_width_hard_block_demand(self):
        for name in ("single_site", "larger"):
            with self.subTest(case=name):
                result = generate_case(self.source, self.work / name, name)
                metadata = result["metadata"]
                lines = result["blif"].read_text().split(".end", 1)[0].splitlines()
                lanes = metadata["circuit"]["lanes"]
                inputs = sum(len(line.split()) - 1 for line in lines if line.startswith(".inputs "))
                outputs = sum(len(line.split()) - 1 for line in lines if line.startswith(".outputs "))
                mult = sum(line.startswith(".subckt multiply ") for line in lines)
                ram = sum(line.startswith(".subckt single_port_ram ") for line in lines)
                luts = sum(line.startswith(".names ") for line in lines)
                latches = sum(line.startswith(".latch ") for line in lines)
                self.assertEqual((inputs, outputs, mult, ram, luts, latches),
                                 (27, lanes, lanes, 40 * lanes, 53 * lanes, lanes))
                self.assertEqual(inputs + outputs + mult + ram + luts + latches,
                                 metadata["circuit"]["atoms_before_vpr_optimization"])
                self.assertGreaterEqual(metadata["capacity_total"]["dsp_top"], lanes)
                self.assertGreaterEqual(metadata["capacity_total"]["memory"], lanes)
                if name == "larger":
                    self.assertEqual(lanes, 128)
                    self.assertEqual(metadata["circuit"]["atoms_before_vpr_optimization"], 12315)

    def test_deterministic_generation_and_refusal_to_overwrite(self):
        first = generate_case(self.source, self.work / "first", "sparse")
        second = generate_case(self.source, self.work / "second", "sparse")
        self.assertEqual(first["arch"].read_bytes(), second["arch"].read_bytes())
        self.assertEqual(first["blif"].read_bytes(), second["blif"].read_bytes())
        self.assertEqual(first["metadata"], second["metadata"])
        before = first["arch"].read_bytes()
        with self.assertRaises(FileExistsError):
            generate_case(self.source, self.work / "first", "sparse")
        self.assertEqual(first["arch"].read_bytes(), before)

    def test_unknown_case_is_rejected_before_output_creation(self):
        target = self.work / "invalid"
        with self.assertRaises(ValueError):
            generate_case(self.source, target, "unknown_case")
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
