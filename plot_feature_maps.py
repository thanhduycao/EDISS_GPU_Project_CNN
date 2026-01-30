import os
import re
import numpy as np
import matplotlib.pyplot as plt


# ---- Configure your three output txt files here ----
FILES = [
    ("Small",  "cpu/pooled_feature_maps_Small.txt"),
    ("Medium", "cpu/pooled_feature_maps_Medium.txt"),
    ("Large",  "cpu/pooled_feature_maps_Large.txt"),
]

# How many maps (channels/kernels) to plot per file (set None to plot all)
MAX_MAPS_PER_FILE = 16

# Output folder
OUT_DIR = "cpu/featuremap_plots"
os.makedirs(OUT_DIR, exist_ok=True)


def _is_header(line: str) -> bool:
    s = line.strip()
    return s.startswith("#")


def _is_blank(line: str) -> bool:
    return line.strip() == ""


def _is_data_line(line: str) -> bool:
    """
    True if the line looks like whitespace-separated floats.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return False
    tok = s.split()[0]
    try:
        float(tok)
        return True
    except ValueError:
        return False


def parse_feature_map_blocks(lines):
    """
    Parse the file into a list of 2D numpy arrays.
    """
    maps = []
    i = 0
    n = len(lines)

    while i < n:
        if _is_blank(lines[i]):
            i += 1
            continue

        if _is_header(lines[i]):
            i += 1
            continue

        if _is_data_line(lines[i]):
            data = []
            while i < n and _is_data_line(lines[i]):
                row = [float(x) for x in lines[i].split()]
                data.append(row)
                i += 1

            arr = np.array(data, dtype=np.float32)

            if arr.ndim == 1:
                arr = arr.reshape(1, -1)

            if arr.dtype == object:
                maxw = max(len(r) for r in data)
                padded = np.full((len(data), maxw), np.nan, dtype=np.float32)
                for r_idx, r in enumerate(data):
                    padded[r_idx, :len(r)] = np.array(r, dtype=np.float32)
                arr = padded

            maps.append(arr)
            continue

        i += 1

    return maps


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def plot_maps(tag, maps, max_maps=None):
    if not maps:
        print(f"[{tag}] No maps found to plot.")
        return

    count = len(maps) if max_maps is None else min(len(maps), max_maps)
    tag_safe = safe_name(tag)

    for idx in range(count):
        fmap = maps[idx]
        h, w = fmap.shape

        plt.figure()
        plt.imshow(fmap, cmap="viridis", aspect="auto")
        plt.colorbar()
        plt.title(f"{tag} - Map {idx} ({h}x{w})")
        plt.axis("off")

        out_path = os.path.join(OUT_DIR, f"{tag_safe}_map_{idx:03d}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"[{tag}] Saved {count} image(s) to: {OUT_DIR}/")


def main():
    for tag, fname in FILES:
        if not os.path.exists(fname):
            print(f"[{tag}] File not found: {fname}")
            continue

        with open(fname, "r") as f:
            lines = f.readlines()

        maps = parse_feature_map_blocks(lines)

        shapes = [m.shape for m in maps[:5]]
        print(f"[{tag}] Parsed {len(maps)} map block(s). First shapes: {shapes}")

        plot_maps(tag, maps, max_maps=MAX_MAPS_PER_FILE)


if __name__ == "__main__":
    main()