"""
VOT-LT2020 sequences, downloaded on demand: only the sequences an experiment actually uses.

The VOT-LT2020 stack uses the 50 VOT-LT2019 (LTB50) sequences. vot-toolkit downloads all of them at once
(17.6 GB); here every sequence is downloaded separately from the same official server, into the same files
vot-toolkit writes (vot/dataset/vot.py, download_dataset_meta):

    <dataset>/<sequence>/groundtruth.txt
    <dataset>/<sequence>/color/00000001.jpg ...
    <dataset>/<sequence>/sequence              (name, fps, format, channels.color)

Every archive is checked against the SHA-1 in the official description. A sequence is extracted into a temporary
folder that is renamed when it is complete, so an interrupted download never leaves half a sequence behind.

Optional cache (`dataset_cache` in configs/paths*.yaml; on Colab a folder on Google Drive): every downloaded
sequence is also stored there as <sequence>.tar, and later sessions restore it from there instead of downloading.
Only the standard library is used.
"""
import csv
import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.parse
import zipfile
from pathlib import Path

from stark_ft import train_data

DESCRIPTION_URL = "https://data.votchallenge.net/vot2019/longterm/description.json"  # vot-toolkit's "vot-lt2019"
# Sequence -> download size in MB (from the official description), in the official order
SEQUENCES = {
    "ballet": 57, "bicycle": 129, "bike1": 322, "bird1": 187, "boat": 1057, "bull": 58, "car1": 314, "car3": 234,
    "car6": 662, "car8": 303, "car9": 251, "car16": 167, "carchase": 92, "cat1": 177, "cat2": 236, "deer": 237,
    "dog": 89, "dragon": 423, "f1": 508, "following": 509, "freesbiedog": 405, "freestyle": 134, "group1": 652,
    "group2": 359, "group3": 931, "helicopter": 444, "horseride": 1756, "kitesurfing": 474, "liverRun": 292,
    "longboard": 501, "nissan": 95, "parachute": 387, "person2": 298, "person4": 329, "person5": 238,
    "person7": 218, "person14": 365, "person17": 424, "person19": 465, "person20": 176, "rollerman": 143,
    "sitcom": 76, "skiing": 147, "sup": 566, "tightrope": 150, "uav1": 158, "volkswagen": 102, "warmup": 828,
    "wingsuit": 343, "yamaha": 123,
}
SMOKE_SEQUENCE = "ballet"  # the smallest sequence (57 MB), used by the smoke test
TMP_DIR = ".download"  # inside the dataset folder: archives being downloaded, sequences being extracted


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_complete(seq_dir: Path) -> bool:
    return (seq_dir / "sequence").is_file() and (seq_dir / "groundtruth.txt").is_file()


def resolve(sequences, dataset_dir: Path) -> list:
    """'all' -> the 50 VOT-LT2020 sequences (sorted); a list is checked against the official names. Sequences that
    are already in the dataset folder are always accepted (own datasets)."""
    if sequences == "all":
        return sorted(SEQUENCES)
    unknown = [s for s in sequences if s not in SEQUENCES and not is_complete(Path(dataset_dir) / s)]
    if unknown:
        raise ValueError(f"Unknown sequence(s): {unknown}\nVOT-LT2020 sequences: {', '.join(SEQUENCES)}")
    return list(sequences)


def missing(names, dataset_dir: Path) -> list:
    return [n for n in names if not is_complete(Path(dataset_dir) / n)]


def size_mb(names) -> int:
    return sum(SEQUENCES.get(n, 0) for n in names)


def size_text(names) -> str:
    mb = size_mb(names)
    return f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb} MB"


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dst_dir: Path, filename: str, checksum: str = None) -> Path:
    path = train_data.download(url, dst_dir, filename, quiet=True)  # resumes an interrupted download
    if checksum and _sha1(path) != checksum:
        path.unlink()
        raise RuntimeError(f"Checksum mismatch for {url}; the file was deleted, run again to download it again.")
    return path


def _write_properties(path: Path, data: dict):
    """The same file as vot.utilities.write_properties (key-sorted 'key=value' lines written by the csv module)."""
    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter="=", escapechar="\\", quoting=csv.QUOTE_NONE).writerows(sorted(data.items()))


def _replace_dir(src: Path, dst: Path):
    if dst.exists():  # only incomplete sequence folders get here (no `sequence` or groundtruth file)
        shutil.rmtree(dst)
    os.replace(src, dst)


def _download_sequence(meta: dict, base_url: str, dataset_dir: Path) -> Path:
    name = meta["name"]
    tmp = dataset_dir / TMP_DIR
    staging = tmp / name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    annotations = meta["annotations"]
    archives = [_fetch(urllib.parse.urljoin(base_url, annotations["url"]), tmp, f"{name}.annotations.zip",
                       annotations.get("checksum"))]
    with zipfile.ZipFile(archives[-1]) as z:
        z.extractall(staging)
    data = {"name": name, "fps": meta["fps"], "format": "default"}
    for cname, channel in meta["channels"].items():
        archives.append(_fetch(urllib.parse.urljoin(base_url, channel["url"]), tmp, f"{name}.{cname}.zip",
                               channel.get("checksum")))
        with zipfile.ZipFile(archives[-1]) as z:
            z.extractall(staging / cname)
        data["channels." + cname] = cname + os.path.sep + channel.get("pattern", "")
    _write_properties(staging / "sequence", data)
    _replace_dir(staging, dataset_dir / name)
    for a in archives:
        a.unlink()
    return dataset_dir / name


def _cache_store(seq_dir: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    part = cache_dir / f"{seq_dir.name}.tar.part"
    with tarfile.open(part, "w") as tar:
        tar.add(str(seq_dir), arcname=seq_dir.name)
    os.replace(part, cache_dir / f"{seq_dir.name}.tar")


def _cache_restore(name: str, dataset_dir: Path, cache_dir: Path) -> bool:
    src = cache_dir / f"{name}.tar"
    if not src.is_file():
        return False
    tmp = dataset_dir / TMP_DIR / "restore"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with tarfile.open(src) as tar:
        tar.extractall(tmp, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    if not is_complete(tmp / name):
        raise RuntimeError(f"The cached archive {src} does not contain the sequence '{name}'. Delete it and run again.")
    _replace_dir(tmp / name, dataset_dir / name)
    shutil.rmtree(tmp)
    return True


def _write_list(dataset_dir: Path):
    """list.txt as vot-toolkit writes it, with the complete sequences (so vot-toolkit can also load the folder)."""
    names = [n for n in SEQUENCES if is_complete(dataset_dir / n)]
    (dataset_dir / "list.txt").write_text("".join(f"{n}\n" for n in names))


def ensure(names, dataset_dir: Path, cache_dir: Path = None) -> list:
    """Makes sure the given sequences are in dataset_dir: restores them from the cache or downloads them.
    Returns the sequences that had to be fetched."""
    dataset_dir = Path(dataset_dir)
    todo = missing(names, dataset_dir)
    if not todo:
        return []
    unknown = [n for n in todo if n not in SEQUENCES]
    if unknown:
        raise ValueError(f"Not in {dataset_dir} and not a VOT-LT2020 sequence: {unknown}")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else None
    restored = [n for n in todo if cache_dir and _cache_restore(n, dataset_dir, cache_dir)]
    if restored:
        _log(f"Restored {len(restored)} sequence(s) from {cache_dir}: {', '.join(restored)}")
    to_download = [n for n in todo if n not in restored]
    if to_download:
        # extracted files + the largest archive (each archive is deleted once its sequence is extracted)
        needed_mb = 1.05 * size_mb(to_download) + max(SEQUENCES[n] for n in to_download)
        free_mb = shutil.disk_usage(dataset_dir).free / 1e6
        if free_mb < needed_mb + 2000:
            raise RuntimeError(f"Not enough disk space for {len(to_download)} VOT sequence(s): about "
                               f"{needed_mb / 1000:.1f} GB (+2 GB margin) needed, {free_mb / 1000:.1f} GB free on "
                               f"{dataset_dir}.")
        _log(f"Downloading {len(to_download)} VOT sequence(s) ({size_text(to_download)}) -> {dataset_dir}")
        tmp = dataset_dir / TMP_DIR
        with open(train_data.download(DESCRIPTION_URL, tmp, "description.json", quiet=True)) as f:
            by_name = {s["name"]: s for s in json.load(f)["sequences"]}
        base_url = DESCRIPTION_URL.rsplit("/", 1)[0] + "/"
        for i, name in enumerate(to_download, 1):
            _log(f"[{i}/{len(to_download)}] {name} ({SEQUENCES[name]} MB)")
            seq_dir = _download_sequence(by_name[name], base_url, dataset_dir)
            if cache_dir:
                _cache_store(seq_dir, cache_dir)
        if cache_dir:
            _log(f"Cached on {cache_dir}: {', '.join(to_download)}")
    shutil.rmtree(dataset_dir / TMP_DIR, ignore_errors=True)
    _write_list(dataset_dir)
    return todo
