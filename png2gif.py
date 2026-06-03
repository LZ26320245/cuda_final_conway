from PIL import Image
import os

folder = "frames"
files = sorted(os.listdir(folder))

images = []

for f in files:
    if f.endswith(".png"):
        images.append(Image.open(os.path.join(folder, f)))

images[0].save(
    "game_of_life.gif",
    save_all=True,
    append_images=images[1:],
    duration=50,  # 每幀 50ms
    loop=0
)

print("GIF created")