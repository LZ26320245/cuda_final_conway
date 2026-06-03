import numpy as np
from PIL import Image
import os

WIDTH = 512
HEIGHT = 512

input_folder = "output"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

files = sorted(os.listdir(input_folder))

for file in files:
    if not file.endswith(".txt"):
        continue

    path = os.path.join(input_folder, file)

    grid = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    with open(path, "r") as f:
        for y, line in enumerate(f):
            line = line.strip()
            for x, c in enumerate(line):
                grid[y, x] = 255 if c == '1' else 0

    img = Image.fromarray(grid, mode="L")
    img.save(os.path.join(output_folder, file.replace(".txt", ".png")))

print("Done: PNG frames generated")