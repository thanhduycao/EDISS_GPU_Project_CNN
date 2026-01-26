import numpy as np
import matplotlib.pyplot as plt

fname = "pooled_feature_maps.txt"
pooled_size = 64
channels_to_read = 4

maps = []
with open(fname, "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("# batch="):
        i += 1
        data = []
        while i < len(lines) and lines[i].strip() != "" and not lines[i].startswith("#"):
            row = [float(x) for x in lines[i].split()]
            data.append(row)
            i += 1
        arr = np.array(data, dtype=np.float32)
        maps.append(arr)
    i += 1

for idx, fmap in enumerate(maps[:channels_to_read]):
    plt.figure()
    plt.imshow(fmap, cmap="viridis")
    plt.colorbar()
    plt.title(f"Pooled Feature Map - Channel {idx}")
    plt.axis("off")
    plt.savefig(f"featuremap_channel_{idx}.png", dpi=200)

print("Saved: featuremap_channel_0.png ... featuremap_channel_3.png")