# e2eET Skeleton Based HGR Using Data-Level Fusion
# Dynamic Hand Gestures Classification: Live Stream Demo
# pyright: reportGeneralTypeIssues=false
# pyright: reportWildcardImportFromLibrary=false
# ---------------------------------------------------------
import shutil
from pathlib import Path

images_directory = Path("../image-datasets")
base_dataset = images_directory / "DHG1428.mVOs-3d.14g-noisy(raw).960px-[allVOs].adaptive-mean"
demo_dataset = images_directory / "DHG1428.liveStream-3D.7G-[custom.topdown.frontaway]"

splits = ["train", "valid"]
gestures = {
    "07-SwipeRight": "SwipeRight",
    "08-SwipeLeft": "SwipeLeft",
    "09-SwipeUp": "SwipeUp",
    "10-SwipeDown": "SwipeDown",
    "11-SwipeX": "SwipeX",
    "12-Swipe+": "Swipe+",
    "13-SwipeV": "SwipeV",
}
excluded_orientations = ["front-to.png", "side-left.png", "side-right.png"]

if not base_dataset.is_dir():
    raise FileNotFoundError(f"Base dataset not found: {base_dataset}")

if demo_dataset.exists():
    shutil.rmtree(demo_dataset)

for split in splits:
    for gesture, label in gestures.items():
        shutil.copytree(
            base_dataset / split / gesture,
            demo_dataset / split / label,
            ignore=shutil.ignore_patterns(*excluded_orientations),
        )

print(f">>> liveStreamHGR dataset created @ {demo_dataset}")
print(f">>> {len(gestures)} gestures | {len(splits)} splits")

for split in splits:
    n_sequences = len(list((demo_dataset / split).glob("*/*")))
    n_images = len(list((demo_dataset / split).rglob("*.png")))
    print(f">>> {split.title()} set [len={n_sequences}, images={n_images}]")
