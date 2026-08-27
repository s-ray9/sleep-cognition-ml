import sys
import shutil
from pathlib import Path

def sort_downloads(target_dir: Path):
    if not target_dir.exists():
        print(f"Error: {target_dir} does not exist.")
        sys.exit(1)

    moved = 0
    skipped = 0

    for fpath in target_dir.glob("*.edf"):
        fname = fpath.name

        if not fname.startswith("sub-"):
            skipped += 1
            continue

        site = fname[4:9]

        if fname.endswith("_expert_annotations.edf"):
            dest_dir = target_dir / "human_annotations" / site
        elif fname.endswith("_caisr_annotations.edf"):
            dest_dir = target_dir / "algorithmic_annotations" / site
        else:
            dest_dir = target_dir / "physiological_data" / site

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / fname

        if dest_path.exists():
            print(f"Already sorted, skipping: {fname}")
            skipped += 1
            continue

        shutil.move(str(fpath), str(dest_path))
        print(f"Moved {fname} -> {dest_path.relative_to(target_dir)}")
        moved += 1

    print(f"\nDone. Moved {moved} files, skipped {skipped}.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sort_downloads.py <target_dir>")
        sys.exit(1)

    sort_downloads(Path(sys.argv[1]))
