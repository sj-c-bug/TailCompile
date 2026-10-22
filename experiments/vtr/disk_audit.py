#!/usr/bin/env python3
"""Read-only, one-pass account disk audit (Python standard library only).

File contents are never opened. Only the explicitly selected output directory is
written. Interior symlinks and filesystem boundaries are not followed.
"""

import argparse
import csv
import heapq
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import threading
import time


def normalize_roots(roots):
    """Resolve existing directory roots and remove duplicates/covered subtrees."""
    resolved = []
    for value in roots:
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise ValueError("Audit root is not a directory: %s" % path)
        if path not in resolved:
            resolved.append(path)
    if not resolved:
        raise ValueError("At least one audit root is required")
    result = []
    for path in resolved:
        if not any(path != other and other in path.parents and
                   path.stat().st_dev == other.stat().st_dev for other in resolved):
            result.append(path)
    return result


def classify_type(path):
    """Classify useful VTR/training artifact families without reading contents."""
    name = Path(path).name.lower()
    if any(token in name for token in ("postroute", "post_route", "post-route", "postrouting", "post_routing", "post-routing")):
        return "post-route artifacts"
    if any(token in name for token in ("rr_graph", "rrgraph", ".rrg")):
        return "routing resource graphs (RRG)"
    if name.endswith((".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".rar", ".gz", ".xz", ".bz2", ".zst")):
        return "archives/compressed"
    if name.endswith((".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5", ".hdf5", ".pkl", ".pickle", ".joblib")):
        return "models/checkpoints"
    if name.endswith((".blif", ".eblif")):
        return "BLIF netlists"
    if name.endswith((".net", ".net.xml")):
        return "NET netlists"
    if name.endswith((".json", ".jsonl", ".ndjson")):
        return "JSON/JSONL"
    if name.endswith((".npy", ".npz", ".parquet", ".arrow", ".feather")):
        return "arrays/datasets"
    if name.endswith((".route", ".place")):
        return "placement/routing"
    if name.endswith((".log", ".out", ".err")):
        return "logs"
    suffix = Path(name).suffix
    return "other " + suffix if suffix else "other (no extension)"


def _human(value):
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(number) < 1024 or unit == "PiB":
            return "%.2f %s" % (number, unit)
        number /= 1024


def _inside(path, root):
    return path == root or root in path.parents


def _allocated(info):
    return getattr(info, "st_blocks", (info.st_size + 511) // 512) * 512


def audit(roots, out, heartbeat_seconds=30, scan_deleted=True):
    """Audit roots; write CSVs/report/progress and return the summary dictionary."""
    requested_roots = [str(p) for p in roots]
    roots = normalize_roots(requested_roots)
    out = Path(out).expanduser().resolve()
    if any(_inside(root, out) for root in roots):
        raise ValueError("Output directory cannot be an audit root or its ancestor")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    progress = {"complete": False, "phase": "metadata", "started_at": started,
                "directories": 0, "files": 0, "allocated_bytes": 0,
                "logical_bytes": 0, "errors": 0, "current_root": None}
    stop = threading.Event()
    write_lock = threading.Lock()
    error_log = (out / "errors.log").open("w", encoding="utf-8", errors="backslashreplace")
    report = (out / "report.log").open("w", encoding="utf-8", errors="backslashreplace")

    def error(message):
        progress["errors"] += 1
        error_log.write(str(message) + "\n")
        error_log.flush()

    def emit(message=""):
        report.write(str(message) + "\n")
        report.flush()

    def save_progress():
        with write_lock:
            snapshot = dict(progress, updated_at=time.time(), elapsed_seconds=round(time.time() - started, 1))
            temporary = out / "progress.json.tmp"
            temporary.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            os.replace(temporary, out / "progress.json")
            return snapshot

    def heartbeat():
        while not stop.wait(max(0.1, heartbeat_seconds)):
            try:
                snapshot = save_progress()
                print(json.dumps(dict(stage="disk_audit_progress", **snapshot),
                                 ensure_ascii=True), flush=True)
            except OSError as exc:
                error("progress write: %s" % exc)

    def command(argv):
        emit("$ " + " ".join(argv))
        if not shutil.which(argv[0]):
            emit("unavailable")
            error("Command unavailable: " + argv[0])
            return
        try:
            result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, errors="replace", timeout=20, check=False)
            emit(result.stdout.rstrip())
            if result.stderr:
                error("%s: %s" % (" ".join(argv), result.stderr.rstrip()))
            if result.returncode:
                error("Command exit %s: %s" % (result.returncode, " ".join(argv)))
            emit("exit=%s" % result.returncode)
        except (OSError, subprocess.TimeoutExpired) as exc:
            error("%s: %s" % (" ".join(argv), exc))
            emit("failed: %s" % exc)

    save_progress()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    print(json.dumps({"stage": "disk_audit_started", "roots": [str(p) for p in roots],
                      "output": str(out)}, ensure_ascii=True), flush=True)
    directories = {}
    root_rows = []
    types = {}
    largest = []
    seen_dirs = set()
    hardlinks = set()
    duplicates = 0
    excluded_filesystems = 0
    excluded_output = 0
    deleted = {"enabled": scan_deleted, "local_only": True, "count": 0,
               "allocated_bytes": 0, "logical_bytes": 0, "permission_errors": 0}
    fatal = None
    scan_errors = 0
    try:
        emit("ACCOUNT DISK AUDIT — metadata only; no file contents read")
        emit("Started: " + time.strftime("%Y-%m-%d %H:%M:%S %z"))
        emit("Requested roots: " + json.dumps(requested_roots, ensure_ascii=True))
        emit("Covered roots: " + json.dumps([str(p) for p in roots], ensure_ascii=True))
        emit("Excluded output: " + str(out))
        emit("Allocated = st_blocks * 512; logical = st_size. Hardlinks counted once globally.")
        emit("Directories include their own metadata blocks and all scanned descendants.")
        emit("Interior symlinks are not followed; each root stays on its initial filesystem.")
        if not hasattr(roots[0].stat(), "st_blocks"):
            emit("WARNING: st_blocks unavailable; allocated bytes are rounded logical estimates.")
        for argv in (["uname", "-a"], ["id"], ["quota", "-s"]):
            command(argv)
        for root in roots:
            for argv in (["df", "-hT", "--", str(root)], ["df", "-i", "--", str(root)],
                         ["findmnt", "-T", str(root)]):
                command(argv)
        progress["phase"] = "scan"
        scan_errors_start = progress["errors"]
        for root in roots:
            progress["current_root"] = str(root)
            root_device = root.stat().st_dev
            root_rows.append({"path": str(root), "device": root_device,
                              "parent": str(root.parent)})
            stack = [(root, None)]
            while stack:
                path, parent = stack.pop()
                try:
                    info = path.lstat()
                    if not stat.S_ISDIR(info.st_mode):
                        error("Directory changed during scan: %s" % path)
                        continue
                    if info.st_dev != root_device:
                        excluded_filesystems += 1
                        continue
                    key = (info.st_dev, info.st_ino)
                    if key in seen_dirs:
                        error("Directory alias already counted: %s" % path)
                        continue
                    seen_dirs.add(key)
                    row = {"path": str(path), "root": str(root), "parent": str(parent) if parent else "",
                           "allocated_bytes": _allocated(info), "logical_bytes": 0,
                           "file_count": 0, "direct_file_count": 0}
                    directories[str(path)] = row
                    progress["directories"] += 1
                    with os.scandir(path) as entries:
                        for entry in entries:
                            child = Path(entry.path)
                            if _inside(child, out):
                                excluded_output += 1
                                continue
                            try:
                                # Windows DirEntry metadata may have zero inode/device
                                # identifiers; lstat supplies the real identifiers.
                                item = os.lstat(entry.path) if os.name == "nt" else entry.stat(follow_symlinks=False)
                                if item.st_dev != root_device:
                                    excluded_filesystems += 1
                                    continue
                                if stat.S_ISDIR(item.st_mode):
                                    stack.append((child, path))
                                    continue
                                unique_key = (item.st_dev, item.st_ino)
                                if item.st_nlink > 1:
                                    if unique_key in hardlinks:
                                        duplicates += 1
                                        continue
                                    hardlinks.add(unique_key)
                                blocks = _allocated(item)
                                row["allocated_bytes"] += blocks
                                if not stat.S_ISREG(item.st_mode):
                                    continue
                                size = item.st_size
                                row["logical_bytes"] += size
                                row["file_count"] += 1
                                row["direct_file_count"] += 1
                                progress["files"] += 1
                                progress["allocated_bytes"] += blocks
                                progress["logical_bytes"] += size
                                kind = classify_type(child)
                                group = types.setdefault(kind, {"type": kind, "file_count": 0,
                                                               "allocated_bytes": 0, "logical_bytes": 0})
                                group["file_count"] += 1
                                group["allocated_bytes"] += blocks
                                group["logical_bytes"] += size
                                candidate = (blocks, size, str(child))
                                if len(largest) < 200:
                                    heapq.heappush(largest, candidate)
                                elif candidate > largest[0]:
                                    heapq.heapreplace(largest, candidate)
                            except OSError as exc:
                                error("stat %s: %s" % (child, exc))
                except OSError as exc:
                    error("scan %s: %s" % (path, exc))
        scan_errors = progress["errors"] - scan_errors_start
        progress["phase"] = "aggregate"
        for row in sorted(directories.values(), key=lambda r: len(Path(r["path"]).parts), reverse=True):
            parent = directories.get(row["parent"])
            if parent:
                for field in ("allocated_bytes", "logical_bytes", "file_count"):
                    parent[field] += row[field]
        ordered_dirs = sorted(directories.values(), key=lambda r: (-r["allocated_bytes"], r["path"]))
        ordered_types = sorted(types.values(), key=lambda r: (-r["allocated_bytes"], r["type"]))
        ordered_files = [{"path": path, "allocated_bytes": blocks, "logical_bytes": size}
                         for blocks, size, path in sorted(largest, reverse=True)]

        def write_csv(name, rows, fields):
            with (out / name).open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        write_csv("directories.csv", ordered_dirs,
                  ["path", "root", "parent", "allocated_bytes", "logical_bytes", "file_count", "direct_file_count"])
        write_csv("file_types.csv", ordered_types, ["type", "file_count", "allocated_bytes", "logical_bytes"])
        write_csv("largest_files.csv", ordered_files, ["path", "allocated_bytes", "logical_bytes"])
        emit("\nROOT TOTALS (allocated / logical / regular file count)")
        for root_row in root_rows:
            row = directories.get(root_row["path"], {})
            root_row.update({key: row.get(key, 0) for key in ("allocated_bytes", "logical_bytes", "file_count")})
            emit("%s / %s / %s  %s [device=%s; parent=%s]" % (
                _human(root_row["allocated_bytes"]), _human(root_row["logical_bytes"]),
                root_row["file_count"], root_row["path"], root_row["device"], root_row["parent"]))
        emit("\nIMMEDIATE CHILD DIRECTORIES (top 100 per root, allocated)")
        for root in roots:
            emit(str(root))
            for row in [r for r in ordered_dirs if r["parent"] == str(root)][:100]:
                emit("%12s  %s" % (_human(row["allocated_bytes"]), json.dumps(row["path"], ensure_ascii=True)))
        emit("\nLARGEST 100 DIRECTORIES (inclusive; overlapping totals must not be added)")
        for row in ordered_dirs[:100]:
            emit("%12s  %s" % (_human(row["allocated_bytes"]), json.dumps(row["path"], ensure_ascii=True)))
        emit("\nFILE TYPES (allocated / logical / count)")
        for row in ordered_types:
            emit("%12s / %12s / %9s  %s" % (_human(row["allocated_bytes"]), _human(row["logical_bytes"]), row["file_count"], row["type"]))
        emit("\nLARGEST 200 FILES (allocated / logical)")
        for row in ordered_files:
            emit("%12s / %12s  %s" % (_human(row["allocated_bytes"]), _human(row["logical_bytes"]), json.dumps(row["path"], ensure_ascii=True)))
        if scan_deleted:
            progress["phase"] = "deleted-open"
            _scan_deleted(roots, deleted, error)
        write_csv("deleted_open.csv", deleted.pop("rows", []),
                  ["pid", "path", "allocated_bytes", "logical_bytes"])
        emit("\nDELETED-OPEN FILES (this host only): " + json.dumps(deleted, sort_keys=True))
        emit("Other GPFS clients can hold deleted-open files; this host's /proc cannot see them.")
        emit("\nINTERPRETATION / UNCLASSIFIED SPACE")
        emit("df describes the filesystem, not necessarily this account or quota. Compare the mount, quota and root totals above.")
        emit("A df-minus-scan gap is not automatically reclaimable: unscanned users/paths, snapshots, deleted-open files,")
        emit("filesystem metadata/reservations, permissions, filesystem boundaries and quota semantics can contribute.")
        emit("This is a live scan, not a snapshot; concurrent changes may cause omissions or double counts.")
        emit("Directory metadata, symlinks and special files contribute to directory totals, not regular-file type totals.")
        scanned_blocks = sum(directories.get(str(r), {}).get("allocated_bytes", 0) for r in roots)
        regular_blocks = sum(v["allocated_bytes"] for v in types.values())
        emit("Scanned allocated: %s; regular files: %s; directory/symlink/special metadata: %s." % (
            _human(scanned_blocks), _human(regular_blocks), _human(scanned_blocks - regular_blocks)))
        emit("Duplicate hardlinks skipped: %s; filesystem-boundary entries skipped: %s; output entries excluded: %s" % (duplicates, excluded_filesystems, excluded_output))
    except BaseException as exc:
        fatal = "%s: %s" % (type(exc).__name__, exc)
        error("Audit interrupted/failed: " + fatal)
    finally:
        stop.set()
        thread.join(timeout=2)
        summary = dict(progress)
        summary.update({"complete": fatal is None, "scan_has_errors": progress["errors"] > 0,
                        "coverage_complete": fatal is None and scan_errors == 0 and excluded_filesystems == 0,
                        "file_counting": "unique regular file inodes; hardlink aliases counted once",
                        "phase": "complete" if fatal is None else "failed",
                        "requested_roots": requested_roots, "roots": root_rows,
                        "output": str(out), "elapsed_seconds": round(time.time() - started, 1),
                        "finished_at": time.time(), "fatal_error": fatal,
                        "duplicate_hardlinks_skipped": duplicates,
                        "skipped_mounts": excluded_filesystems,
                        "filesystem_entries_skipped": excluded_filesystems,
                        "deleted_open": deleted,
                        "file_allocated_bytes": sum(v["allocated_bytes"] for v in types.values()),
                        "directory_allocated_bytes": sum(directories.get(str(r), {}).get("allocated_bytes", 0) for r in roots)})
        summary["allocated_bytes"] = summary["directory_allocated_bytes"]
        summary["directory_and_nonregular_allocated_bytes"] = summary["allocated_bytes"] - summary["file_allocated_bytes"]
        progress.update(summary)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        save_progress()
        emit("\nFinished: complete=%s, errors=%s, elapsed=%.1fs" % (summary["complete"], summary["errors"], summary["elapsed_seconds"]))
        emit("Share report.log, summary.json, file_types.csv and largest_files.csv. Full directory inventory: directories.csv.")
        report.close()
        error_log.close()
    return summary


def _scan_deleted(roots, result, error):
    """Count matching unlinked open files; never report other processes' paths."""
    proc = Path("/proc")
    if not proc.is_dir():
        result["available"] = False
        return
    result["available"] = True
    result["rows"] = []
    seen = set()
    try:
        processes = list(proc.iterdir())
    except OSError as exc:
        error("Cannot list /proc: %s" % exc)
        return
    for process in processes:
        if not process.name.isdigit():
            continue
        try:
            with os.scandir(process / "fd") as entries:
                for entry in entries:
                    try:
                        target = os.readlink(entry.path)
                        if not target.endswith(" (deleted)"):
                            continue
                        target_path = Path(target[:-10])
                        if not target_path.is_absolute() or not any(_inside(target_path, root) for root in roots):
                            continue
                        info = os.stat(entry.path)
                        key = (info.st_dev, info.st_ino)
                        if info.st_nlink == 0 and stat.S_ISREG(info.st_mode) and key not in seen:
                            seen.add(key)
                            result["count"] += 1
                            result["allocated_bytes"] += _allocated(info)
                            result["logical_bytes"] += info.st_size
                            result["rows"].append({"pid": process.name, "path": str(target_path),
                                                   "allocated_bytes": _allocated(info), "logical_bytes": info.st_size})
                    except PermissionError:
                        result["permission_errors"] += 1
                    except OSError:
                        pass  # Process/fd may disappear during this live scan.
        except PermissionError:
            result["permission_errors"] += 1
        except OSError:
            pass
    if result["permission_errors"]:
        error("Deleted-open scan has restricted /proc access (%s permission errors); count is a lower bound." % result["permission_errors"])


def run_audit(args):
    return audit([args.root] + (args.extra_root or []), args.out,
                 heartbeat_seconds=getattr(args, "heartbeat_seconds", 30),
                 scan_deleted=not getattr(args, "no_deleted_open", False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Explicit account directory to audit")
    parser.add_argument("--extra-root", action="append", default=[], help="Additional repo/scratch/conda root; repeatable")
    parser.add_argument("--out", required=True, help="New output directory inside the user's workspace, e.g. repo/logs/storage/account_audit")
    parser.add_argument("--heartbeat-seconds", type=float, default=30)
    parser.add_argument("--no-deleted-open", action="store_true", help="Skip local /proc deleted-open check")
    args = parser.parse_args()
    try:
        result = run_audit(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("DISK_AUDIT_%s: %s" % ("COMPLETE" if result["complete"] else "FAILED", Path(args.out).resolve()))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
