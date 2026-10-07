"""
Preparing the training datasets in the layout expected by lib/train/dataset:

    <train_data>/coco/annotations/instances_train2017.json
    <train_data>/coco/images/train2017/*.jpg                      (118287 images)
    <train_data>/got10k/train/list.txt
    <train_data>/got10k/train/GOT-10k_Train_000001/ ...           (9335 sequences)

COCO 2017 is downloaded from the official server (images.cocodataset.org). GOT-10k needs a (free) registration, so it
is prepared from the official archives (stark_ft/got10k.py). The archives are kept in the archive folder (paths key
`archives`, default <train_data>/_archives; on Colab a folder on Google Drive), so another machine / session only has
to extract them. A dataset folder that already has the expected layout is only read, never modified.
"""
import os
import shutil
from pathlib import Path

from stark_ft import got10k
from stark_ft.archives import TMP_NAME, check_space, extract, uncompressed_size
from stark_ft.download import _log, download

COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
COCO_TRAIN_IMAGES = 118287


def coco_ready(root: Path) -> bool:
    return (root / "annotations" / "instances_train2017.json").is_file() and (root / "images" / "train2017").is_dir()


# ---------------------------------------------------------------------- COCO
def prepare_coco(root: Path, archive_dir: Path, delete_archives=False) -> Path:
    if coco_ready(root):
        _log(f"COCO ready: {root}")
        return root
    if root.exists() and any(p.name != TMP_NAME for p in root.iterdir()):
        raise RuntimeError(f"{root} exists but does not have the expected COCO layout "
                           "(annotations/instances_train2017.json, images/train2017/). Move it away or fix it.")
    archives = {name: download(url, archive_dir, name) for name, url in COCO_URLS.items()}
    tmp = root / TMP_NAME
    if tmp.exists():
        shutil.rmtree(tmp)
    check_space(root, sum(uncompressed_size(a) for a in archives.values()), "COCO train2017")
    try:
        extract(archives["annotations_trainval2017.zip"], tmp)     # -> annotations/
        extract(archives["train2017.zip"], tmp / "images")          # -> images/train2017/
        n = sum(1 for _ in os.scandir(tmp / "images" / "train2017"))
        if n != COCO_TRAIN_IMAGES:
            raise RuntimeError(f"train2017.zip produced {n} images instead of {COCO_TRAIN_IMAGES}; delete "
                               f"{archives['train2017.zip']} and run the preparation again.")
        for member in ("annotations", "images"):
            os.replace(tmp / member, root / member)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if delete_archives:
        for a in archives.values():
            a.unlink()
    _log(f"COCO ready: {root} ({n} images)")
    return root


# ---------------------------------------------------------------------- entry point
SUPPORTED = ("got10k", "coco")
ALIASES = {"got10k_full": "got10k"}  # the same folder


def prepare(datasets, train_data: Path, archive_dir: Path = None, got10k_urls=(), delete_archives=False) -> dict:
    """Prepares the given datasets ("got10k", "coco") under train_data. Returns {dataset: folder}."""
    train_data = Path(train_data)
    archive_dir = Path(archive_dir) if archive_dir else train_data / "_archives"
    datasets = sorted({ALIASES.get(d, d) for d in datasets})
    unsupported = [d for d in datasets if d not in SUPPORTED]
    if unsupported:
        raise ValueError(f"Automatic preparation supports {list(SUPPORTED)}; prepare {unsupported} manually "
                         "(see docs/train.md).")
    out = {}
    if "coco" in datasets:
        out["coco"] = str(prepare_coco(train_data / "coco", archive_dir / "coco", delete_archives))
    if "got10k" in datasets:
        out["got10k"] = str(got10k.prepare_split("train", train_data / "got10k", archive_dir / "got10k", got10k_urls,
                                                 delete_archives))
    return out
