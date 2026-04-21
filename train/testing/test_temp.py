import json
from pathlib import Path
from PIL import Image

scene = Path("/home/jake/projects/stable-virtual-camera/dataset/dl3dv_parsed/11K/scenes/000000")
tf = json.load(open(scene / "transforms.json"))

fr = tf["frames"][0]
img_path = scene / fr["file_path"]

with Image.open(img_path) as img:
    actual_w, actual_h = img.size

print("actual image size:", actual_w, actual_h)

print("\nTop-level metadata:")
for k in ["w", "h", "fl_x", "fl_y", "cx", "cy", "camera_model", "k1", "k2", "p1", "p2"]:
    print(k, tf.get(k))

print("\nFrame-level metadata:")
for k in ["w", "h", "fl_x", "fl_y", "cx", "cy", "camera_model", "k1", "k2", "p1", "p2"]:
    print(k, fr.get(k))

fl_x = tf.get("fl_x", fr.get("fl_x"))
fl_y = tf.get("fl_y", fr.get("fl_y"))
cx = tf.get("cx", fr.get("cx"))
cy = tf.get("cy", fr.get("cy"))

print("\nNormalized by actual image size:")
print("fl_x / actual_w:", fl_x / actual_w)
print("fl_y / actual_h:", fl_y / actual_h)
print("cx / actual_w:", cx / actual_w)
print("cy / actual_h:", cy / actual_h)
