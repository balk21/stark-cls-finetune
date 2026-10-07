"""
GOT-10k (http://got-10k.aitestunion.com): the train / val / test splits, prepared from the official archives.

    <root>/train/   list.txt, GOT-10k_Train_000001/ ... GOT-10k_Train_009335/   training, and testing
    <root>/val/     list.txt, GOT-10k_Val_000001/  ... GOT-10k_Val_000180/      testing (ground truth available)
    <root>/test/    list.txt, GOT-10k_Test_000001/ ... GOT-10k_Test_000180/     testing (first-frame box only;
                                                                                 scored by the GOT-10k server)
<root> is <train_data>/got10k. GOT-10k can only be downloaded after a (free) registration, so the splits are prepared
from the official archives (e.g. full_data.zip, which contains train/, val/ and test/, or the split archives;
archives inside archives are extracted as well), found in <archives>/got10k/ or given as download links / paths.
Only the members of the requested split are extracted. Extraction goes into a temporary folder that is renamed when
complete; a split folder that already has list.txt is only read, never modified.
"""
import os
import re
import shutil
from pathlib import Path

from stark_ft import archives as arc
from stark_ft.download import _log, download

SPLITS = {"train": ("Train", 9335), "val": ("Val", 180), "test": ("Test", 180)}  # split -> (name tag, sequences)


def all_sequences(split: str):
    tag, n = SPLITS[split]
    return [f"GOT-10k_{tag}_{i:06d}" for i in range(1, n + 1)]


def members(split: str):
    """Wildcard patterns of the archive members that belong to a split (incl. archives inside archives)."""
    tag = SPLITS[split][0]
    nested = tuple(f"*{t}*{s}" for t in (tag, tag.lower()) for s in arc.ARCHIVE_SUFFIXES)
    return (f"*GOT-10k_{tag}_*", "*list.txt") + nested


def ready(split_dir: Path) -> bool:
    return (Path(split_dir) / "list.txt").is_file()


def _walk(root: Path, split: str):
    """(archives, {sequence name: folder}) under root, without entering sequence folders (many image files)."""
    pattern = re.compile(rf"^GOT-10k_{SPLITS[split][0]}_\d{{6}}$")
    sequences = {}

    def is_sequence(dirpath, d):
        if pattern.match(d) and os.path.isfile(os.path.join(dirpath, d, "groundtruth.txt")):
            sequences.setdefault(d, Path(dirpath) / d)
            return True
        return False

    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != arc.TMP_NAME and not is_sequence(dirpath, d)]
        found += [Path(dirpath) / f for f in filenames if arc.is_archive(f)]
    return sorted(found), sequences


def _gather(archive_dir: Path, urls):
    found = []
    for src in urls:  # download links, or archives / folders that are used in place
        path = Path(os.path.expanduser(str(src)))
        if re.match(r"^(https?|ftp)://", str(src)):
            download(str(src), archive_dir)
        elif path.is_dir():
            found += arc.find_archives(path)
        elif path.is_file() and arc.is_archive(path.name):
            found.append(path)
        else:
            raise FileNotFoundError(f"GOT-10k source {src!r} is neither a URL nor an existing archive / folder")
    return sorted(set(found + (arc.find_archives(archive_dir) if archive_dir.is_dir() else [])))


def prepare_split(split: str, root: Path, archive_dir: Path, urls=(), delete_archives=False) -> Path:
    """Makes sure <root>/<split> exists (extracts it from the GOT-10k archives if needed). Returns the folder."""
    tag, expected = SPLITS[split]
    split_dir = Path(root) / split
    if ready(split_dir):
        _log(f"GOT-10k {split} ready: {split_dir}")
        return split_dir
    if split_dir.exists():
        raise RuntimeError(f"{split_dir} exists but has no list.txt (not a complete GOT-10k {split} folder). "
                           "Move it away or fix it.")
    archive_dir = Path(archive_dir)
    found = _gather(archive_dir, urls)
    if not found:
        archive_dir.mkdir(parents=True, exist_ok=True)  # so the folder to put the archives into exists
        raise FileNotFoundError(
            f"No GOT-10k archive in {archive_dir} (the folder exists now, it is empty).\n"
            "Put the GOT-10k archives (e.g. the official full_data.zip, which contains train, val and test) into this "
            "folder and run again. With a Google Drive link: open it in the browser -> 'Add shortcut to Drive' -> "
            "right-click the shortcut -> 'Make a copy' -> move the copy ('Copy of full_data.zip') into this folder "
            "(docs/colab.md).\nThe links come by e-mail after a free registration at "
            "http://got-10k.aitestunion.com/downloads ; download links or archive paths can also be given with "
            "--got10k-url (notebook: got10k_sources).")
    found = arc.without_duplicates(found)
    wanted = members(split)
    tmp = split_dir.parent / f"{arc.TMP_NAME}_{split}"
    if tmp.exists():
        shutil.rmtree(tmp)
    arc.check_space(tmp, sum(arc.uncompressed_size(a, wanted) for a in found), f"GOT-10k {split}")
    try:
        for a in found:
            arc.extract(a, tmp / "extracted", wanted)
        # Archives inside archives (e.g. .../GOT-10k_Train_split_01.zip): extract them one at a time and delete each
        # (temporary) copy right away to limit the disk usage.
        nested, sequences = _walk(tmp / "extracted", split)
        while nested:
            for a in nested:
                arc.check_space(tmp, arc.uncompressed_size(a, wanted), a.name)
                arc.extract(a, a.parent, wanted)
                a.unlink()
            nested, sequences = _walk(tmp / "extracted", split)
        if len(sequences) != expected:
            raise RuntimeError(f"The GOT-10k archives contain {len(sequences)} {split} sequences; all {expected} are "
                               f"needed (archives in {archive_dir}).")
        out = tmp / split
        out.mkdir()
        for name, d in sequences.items():
            os.replace(d, out / name)
        names = sorted(sequences)
        # The official list.txt if the archives contain it, otherwise an identical one (sorted, no trailing newline)
        official = [Path(dp) / "list.txt" for dp, _, files in os.walk(tmp / "extracted") if "list.txt" in files]
        official = [p for p in official if p.read_text().split() == names]
        if official:
            shutil.copy2(official[0], out / "list.txt")
        else:
            (out / "list.txt").write_text("\n".join(names))
        os.replace(out, split_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if delete_archives:  # only the copies in the archive folder, never archives given by path
        for a in found:
            if archive_dir.resolve() in a.resolve().parents:
                a.unlink()
    _log(f"GOT-10k {split} ready: {split_dir} ({len(names)} sequences)")
    return split_dir


def read_sequence(split_dir: Path, name: str):
    """(image paths, ground-truth boxes with None where the target is not visible, (width, height), first-frame box).
    The test split has the box of the first frame only; the other frames are None."""
    import cv2
    seq_dir = Path(split_dir) / name
    images = sorted(seq_dir.glob("*.jpg"))
    gt = []
    with open(seq_dir / "groundtruth.txt") as f:
        for line in f:
            if line.strip():
                gt.append([float(v) for v in line.replace(" ", "").split(",")[:4]])
    visible = [True] * len(gt)
    cover = seq_dir / "cover.label"  # GOT-10k evaluates the frames with cover > 0 (target visible)
    if cover.is_file():
        visible = [int(v) > 0 for v in cover.read_text().split()][:len(gt)]
    boxes = [g if v else None for g, v in zip(gt, visible)]
    boxes += [None] * (len(images) - len(boxes))
    h, w = cv2.imread(str(images[0])).shape[:2]
    return images, boxes, (w, h), gt[0]
