import json
import os
from datetime import datetime
from pathlib import Path


def dedupe_region_bounds(region_dir: Path):
    # Group all json after bounds
    groups = {}

    for path in sorted(region_dir.rglob("*.json")):
        if not path.exists():
            continue

        meta = json.load(open(path))
        bounds = tuple(meta["bounds"])
        timestamp = datetime.fromisoformat(meta["timestamp"])

        groups.setdefault(bounds, []).append((timestamp, path))

    # find groups with more than one file
    duplicates = {b: files for b, files in groups.items() if len(files) > 1}

    print("\nFound duplicate groups:\n")
    for bounds, files in duplicates.items():
        print("BOUNDS:", bounds)
        for ts, p in sorted(files):
            print("  ", ts, " -> ", p)
        print()

    if not duplicates:
        print("No duplicates found.")
        return []

    ans = input("Remove older files? [y/N]: ").strip().lower()
    if ans != "y":
        print("Aborted. No files removed.")
        return []

    removed = []

    # remove all but the newest
    for bounds, files in duplicates.items():
        files_sorted = sorted(files)  # sort by timestamp
        keep_ts, keep_path = files_sorted[-1]  # newest
        to_remove = files_sorted[:-1]  # all older

        for ts, p in to_remove:
            if p.exists():
                p.unlink()
                removed.append(p)

    print("\nRemoved:")
    for r in removed:
        print("  ", r)

    return removed


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT"))
    region_dir = PROJECT_ROOT / "dataset" / "region"

    print("region dir:", region_dir)
    print("region dir exists:", region_dir.exists())

    dedupe_region_bounds(region_dir)
