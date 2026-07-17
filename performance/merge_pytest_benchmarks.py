#!/usr/bin/env python3
"""Merge several pytest-benchmark result files for one save-name into a single file.

    python merge_pytest_benchmarks.py <storage_platform_dir> <save_name>
    e.g. python merge_pytest_benchmarks.py benchmarks/Darwin-CPython-3.12-64bit BiCM_large

Why this exists: the harness sometimes runs one problem's benchmarks as SEVERAL pytest
invocations (e.g. the BiCM large projection variants each run in their own process so a
per-job timeout can kill one variant without losing the others). pytest-benchmark then
writes one ``NNNN_<save_name>.json`` per invocation, but the plotting scripts'
``find_latest_files`` keeps only the newest file per scale, so without a merge the last
invocation would silently shadow all the earlier ones.

This unions the ``benchmarks`` arrays of every ``NNNN_<save_name>.json`` in the storage
directory (deduplicated by test ``name``, the newest file winning) and writes the result
as a new highest-numbered file, which the plotters then pick up as the newest file for
that scale. Jobs that were killed before saving simply are not present. Uses only the
standard library.
"""
import json
import platform
import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    storage, save_name = Path(sys.argv[1]), sys.argv[2]
    if not storage.is_dir():
        print(f"merge_pytest_benchmarks: no such directory: {storage}", file=sys.stderr)
        return 2

    pattern = re.compile(r"^(\d+)_" + re.escape(save_name) + r"\.json$")

    def matches(d: Path):
        return sorted(
            (p for p in d.iterdir() if p.is_file() and pattern.match(p.name)),
            key=lambda p: p.stat().st_mtime,
        )

    # The caller may pass either a platform directory (e.g. .../Darwin-CPython-3.12-64bit) or the
    # base storage directory that pytest-benchmark puts the platform directories in. In the latter
    # case, resolve to THIS interpreter's platform directory (this script runs inside the same
    # venv that just produced the files, so the two names agree by construction); results from
    # different machines/toolchains must never be merged together. Fall back to the directory
    # holding the most recent matching file only if the constructed name has no matches.
    parts = matches(storage)
    if not parts:
        me = "%s-%s-%s-%sbit" % (
            platform.system(),
            platform.python_implementation(),
            ".".join(platform.python_version_tuple()[:2]),
            64 if sys.maxsize > 2**32 else 32,
        )
        if (storage / me).is_dir() and matches(storage / me):
            storage = storage / me
            parts = matches(storage)
            print(f"merge_pytest_benchmarks: using platform directory {storage.name}")
        else:
            candidates = [(d, matches(d)) for d in storage.iterdir() if d.is_dir()]
            candidates = [(d, m) for d, m in candidates if m]
            if candidates:
                storage, parts = max(candidates, key=lambda dm: dm[1][-1].stat().st_mtime)
                print(f"merge_pytest_benchmarks: using platform directory {storage.name} (newest match)")
    if not parts:
        print(f"merge_pytest_benchmarks: nothing matching *_{save_name}.json in {storage}", file=sys.stderr)
        return 1
    if len(parts) == 1:
        print(f"merge_pytest_benchmarks: only {parts[0].name}, nothing to merge")
        return 0

    # Oldest first, so later (newer) files overwrite earlier entries of the same test name.
    merged_by_name = {}
    header = None
    for p in parts:
        data = json.loads(p.read_text())
        header = data  # keep the newest file's machine_info/commit_info/datetime/version
        for bench in data.get("benchmarks", []):
            merged_by_name[bench["name"]] = bench

    out = {k: v for k, v in header.items() if k != "benchmarks"}
    out["benchmarks"] = list(merged_by_name.values())

    # Next global counter in this storage dir, so the merged file is the newest for its scale
    # both by counter and by mtime.
    counters = [
        int(m.group(1))
        for p in storage.iterdir()
        if (m := re.match(r"^(\d+)_", p.name))
    ]
    next_counter = max(counters, default=0) + 1
    dest = storage / f"{next_counter:04d}_{save_name}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(
        f"merge_pytest_benchmarks: merged {len(parts)} files "
        f"({sum(1 for _ in merged_by_name)} distinct benchmarks) -> {dest.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
