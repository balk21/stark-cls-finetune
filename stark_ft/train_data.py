"""
Preparing the training datasets in the layout expected by lib/train/dataset:

    <train_data>/coco/annotations/instances_train2017.json
    <train_data>/coco/images/train2017/*.jpg                      (118287 images)
    <train_data>/got10k/train/list.txt
    <train_data>/got10k/train/GOT-10k_Train_000001/ ...           (9335 sequences)

COCO 2017 is downloaded from the official server (images.cocodataset.org). GOT-10k can only be downloaded after a
(free) registration on http://got-10k.aitestunion.com/downloads, which sends the download links by e-mail, so it is
prepared from the official archives (e.g. full_data.zip, or the train split zips; archives inside archives are
extracted as well), given in `got10k_urls` as
  - download links (Google Drive share links such as https://drive.google.com/file/d/<id>/view are converted to
    direct downloads), or
  - paths of archives / folders, used in place (e.g. a shortcut on the mounted Google Drive in Colab),
or put into <archives>/got10k/. Only the GOT-10k train videos (and list.txt files) are extracted from them.

The archives are kept in an archive folder (default <train_data>/_archives; on Colab a folder on Google Drive), so
another machine / a new Colab session only has to extract them. Extraction goes into a temporary folder that is
renamed when it is complete, so an interrupted preparation never leaves a half-extracted dataset behind. A dataset
folder that already has the expected layout is only read, never modified.
"""
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

COCO_URLS = {
    "train2017.zip": "http://images.cocodataset.org/zips/train2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}
COCO_TRAIN_IMAGES = 118287
GOT10K_TRAIN_SEQUENCES = 9335  # lib/train/data_specs/got10k_vot_*_split.txt index into this (sorted) list
GOT10K_SEQ = re.compile(r"^GOT-10k_Train_\d{6}$")
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
TMP_NAME = ".stark_prepare_tmp"
URL_MAP = ".downloads.json"  # url -> file name of finished downloads (e-mailed links may expire)
# Members extracted from GOT-10k archives: train videos, list.txt files and archives inside archives
GOT10K_MEMBERS = ("*GOT-10k_Train_*", "*list.txt") + tuple("*" + s for s in ARCHIVE_SUFFIXES)
_DRIVE_ID = re.compile(r"(?:drive\.google\.com/(?:file/d/|open\?(?:[^#]*&)?id=|uc\?(?:[^#]*&)?id=)"
                       r"|drive\.usercontent\.google\.com/download\?(?:[^#]*&)?id=)([\w-]{20,})")
FREE_MARGIN = 2e9


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _gb(n):
    return f"{n / 1e9:.1f} GB"


# ---------------------------------------------------------------------- layout checks (read-only)
def coco_ready(root: Path) -> bool:
    return (root / "annotations" / "instances_train2017.json").is_file() and (root / "images" / "train2017").is_dir()


def got10k_ready(train_dir: Path) -> bool:
    return (train_dir / "list.txt").is_file()


def _walk(root: Path):
    """Returns (archives, GOT-10k train sequence folders) under root without descending into sequence folders
    (~1.4 million image files)."""
    archives, sequences = [], {}
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in dirnames:
            if GOT10K_SEQ.match(d) and os.path.isfile(os.path.join(dirpath, d, "groundtruth.txt")):
                sequences.setdefault(d, Path(dirpath) / d)
            elif d != TMP_NAME:
                keep.append(d)
        dirnames[:] = keep
        archives += [Path(dirpath) / f for f in filenames if f.lower().endswith(ARCHIVE_SUFFIXES)]
    return sorted(archives), sequences


# ---------------------------------------------------------------------- download
def _filename_from_response(url, response):
    disposition = response.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", disposition)
    if m:
        return Path(urllib.parse.unquote(m.group(1))).name
    return Path(urllib.parse.unquote(urllib.parse.urlparse(response.geturl()).path)).name


def direct_url(url: str) -> str:
    """Google Drive share links ('.../file/d/<id>/view', 'open?id=<id>', 'uc?id=<id>') -> direct download link
    (without the "cannot scan for viruses" confirmation page). Other URLs are returned unchanged."""
    m = _DRIVE_ID.search(url)
    return f"https://drive.usercontent.google.com/download?id={m.group(1)}&export=download&confirm=t" if m else url


QUOTA_HELP = ("Google Drive limits how often a shared file can be downloaded per day, and this file's limit is "
              "used up (it may work again later). The reliable way: open the link in the browser, 'Add shortcut to "
              "Drive', right-click the shortcut -> 'Make a copy', and put the copy into the GOT-10k archive folder (on "
              "Colab <DRIVE_ROOT>/train_archives/got10k/). Your own copy has no such limit. See README, section 8.3.")


def _open(url, headers=None):
    r = urllib.request.urlopen(urllib.request.Request(direct_url(url), headers=headers or {}), timeout=60)
    if r.headers.get("Content-Type", "").startswith("text/html"):
        page = r.read(20000).decode("utf-8", errors="replace")
        r.close()
        title = re.search(r"<title>(.*?)</title>", page, re.S)
        title = title.group(1).strip() if title else ""
        if "quota" in title.lower():
            raise RuntimeError(f"{url}: '{title}'.\n{QUOTA_HELP}")
        raise RuntimeError(
            f"{url} returned a web page ('{title}') instead of a file. Use a direct download link; a Google Drive file "
            "must be shared with 'anyone with the link'.")
    return r


def download(url: str, dst_dir: Path, filename: str = None, quiet: bool = False) -> Path:
    """Downloads `url` into dst_dir, resuming an interrupted download (<file>.part). Returns the file path.
    quiet=True: no start line (progress is still shown for long downloads)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    map_file = dst_dir / URL_MAP
    url_map = json.loads(map_file.read_text()) if map_file.is_file() else {}
    filename = filename or url_map.get(url)
    if filename and (dst_dir / filename).is_file():
        return dst_dir / filename

    r = _open(url)
    try:
        filename = filename or _filename_from_response(url, r)
        if not filename:
            raise RuntimeError(f"Cannot determine a file name for {url}")
        dst, part = dst_dir / filename, dst_dir / (filename + ".part")
        if not dst.is_file():
            total = int(r.headers.get("Content-Length") or 0)
            pos = part.stat().st_size if part.is_file() else 0
            if pos:
                r.close()
                r = _open(url, {"Range": f"bytes={pos}-"})
                if r.status != 206:  # the server ignored the Range header: start again
                    pos = 0
            if not quiet or pos:
                _log(f"Downloading {filename} ({_gb(total) if total else 'unknown size'})"
                     f"{f', resuming at {_gb(pos)}' if pos else ''} -> {dst_dir}")
            done, t0, last = pos, time.time(), time.time()
            with open(part, "ab" if pos else "wb") as f:
                for chunk in iter(lambda: r.read(8 << 20), b""):
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - last > 60:
                        last = time.time()
                        rate = (done - pos) / max(last - t0, 1e-9)
                        eta = f", ETA {(total - done) / rate / 60:.0f} min" if total and rate > 0 else ""
                        _log(f"  {_gb(done)}{f' / {_gb(total)}' if total else ''} ({rate / 1e6:.0f} MB/s{eta})")
            if total and part.stat().st_size != total:
                raise RuntimeError(f"Incomplete download of {filename}: {part.stat().st_size} of {total} bytes. "
                                   "Run the preparation again to resume.")
            os.replace(part, dst)
    finally:
        r.close()
    url_map[url] = filename
    map_file.write_text(json.dumps(url_map, indent=1))
    return dst


# ---------------------------------------------------------------------- extraction
def _selected(name: str, members) -> bool:
    return members is None or any(fnmatch.fnmatch(name, pattern) for pattern in members)


def _read_error(archive: Path, err) -> RuntimeError:
    msg = f"Cannot read {archive}: {err}"
    if "/drive/" in str(archive):  # Google Drive mount (Colab)
        msg += ("\nIf this is a shortcut to a file shared by someone else, Google Drive probably refuses to serve it "
                "because the file's download quota is used up.\n" + QUOTA_HELP)
    else:
        msg += "\nIf it is an incomplete download, delete it and run the preparation again."
    return RuntimeError(msg)


def _uncompressed_size(archive: Path, members=None) -> int:
    if archive.name.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(archive) as z:
                return sum(i.file_size for i in z.infolist() if _selected(i.filename, members))
        except (zipfile.BadZipFile, OSError) as e:
            raise _read_error(archive, e) from None
    return archive.stat().st_size * (1 if archive.name.lower().endswith(".tar") else 2)


def _without_duplicates(archives):
    """The same archive twice, e.g. your copy and a shortcut of full_data.zip on Google Drive, is used only once.
    Zips count as the same if size, member names and CRCs match (only the zip's directory is read)."""
    by_size = {}
    for a in archives:
        by_size.setdefault(a.stat().st_size, []).append(a)
    kept, seen = [], {}
    for a in archives:
        key = a
        if len(by_size[a.stat().st_size]) > 1 and a.name.lower().endswith(".zip"):
            h = hashlib.sha1()
            try:
                with zipfile.ZipFile(a) as z:
                    for i in z.infolist():
                        h.update(f"{i.filename}\0{i.CRC}\n".encode())
            except (zipfile.BadZipFile, OSError) as e:
                raise _read_error(a, e) from None
            key = (a.stat().st_size, h.hexdigest())
        if key in seen:
            _log(f"Skipping {a.name}: the same archive as {seen[key].name} (one of them can be removed from "
                 f"{a.parent})")
            continue
        seen[key] = a
        kept.append(a)
    return kept


def _check_space(where: Path, needed: int, what: str):
    probe = where
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed + FREE_MARGIN:
        raise RuntimeError(f"Not enough disk space for {what}: {_gb(needed)} needed (+{_gb(FREE_MARGIN)} margin), "
                           f"{_gb(free)} free on {probe}. Use a machine / Colab runtime with a larger disk, or train "
                           "on fewer datasets.")


def extract(archive: Path, dst: Path, members=None):
    """Extracts the archive (only the members matching the wildcard patterns `members`, if given) into dst."""
    dst.mkdir(parents=True, exist_ok=True)
    _log(f"Extracting {archive.name} ({_gb(archive.stat().st_size)}) ...")
    if archive.name.lower().endswith(".zip"):
        if members is not None:  # only patterns that are the first match of some member (unzip warns otherwise)
            with zipfile.ZipFile(archive) as z:
                names = z.namelist()
            regexes, used = [re.compile(fnmatch.translate(m)) for m in members], set()
            for name in names:
                first = next((i for i, r in enumerate(regexes) if r.match(name)), None)
                if first is not None:
                    used.add(first)
                    if len(used) == len(regexes):
                        break
            members = [m for i, m in enumerate(members) if i in used]
            if not members:
                return
        if shutil.which("unzip"):
            rc = subprocess.run(["unzip", "-q", "-o", str(archive), *(members or ()), "-d", str(dst)]).returncode
            if rc not in (0, 1, 11):  # 1 = warnings only, 11 = no matching members
                raise _read_error(archive, f"unzip failed (exit code {rc})")
        else:
            with zipfile.ZipFile(archive) as z:
                z.extractall(dst, [n for n in z.namelist() if _selected(n, members)])
    else:
        with tarfile.open(archive) as t:
            t.extractall(dst, [m for m in t.getmembers() if _selected(m.name, members)])


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
    _check_space(root, sum(_uncompressed_size(a) for a in archives.values()), "COCO train2017")
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


# ---------------------------------------------------------------------- GOT-10k
def prepare_got10k(train_dir: Path, archive_dir: Path, urls=(), delete_archives=False) -> Path:
    if got10k_ready(train_dir):
        _log(f"GOT-10k ready: {train_dir}")
        return train_dir
    if train_dir.exists():
        raise RuntimeError(f"{train_dir} exists but has no list.txt (not a complete GOT-10k train folder). "
                           "Move it away or fix it.")
    archives = []
    for src in urls:  # download links, or archives / folders that are used in place
        path = Path(os.path.expanduser(str(src)))
        if re.match(r"^(https?|ftp)://", str(src)):
            download(str(src), archive_dir)
        elif path.is_dir():
            archives += _walk(path)[0]
        elif path.is_file() and path.name.lower().endswith(ARCHIVE_SUFFIXES):
            archives.append(path)
        else:
            raise FileNotFoundError(f"GOT-10k source {src!r} is neither a URL nor an existing archive / folder")
    archives = sorted(set(archives + (_walk(archive_dir)[0] if archive_dir.is_dir() else [])))
    if not archives:
        archive_dir.mkdir(parents=True, exist_ok=True)  # so the folder to put the archives into exists
        raise FileNotFoundError(
            f"No GOT-10k archive in {archive_dir} (the folder exists now, it is empty).\n"
            "Put the GOT-10k train archives (e.g. the official full_data.zip) into this folder and run again. "
            "On Colab with a Google Drive link: open the link in the browser -> 'Add shortcut to Drive' -> "
            "right-click the shortcut -> 'Make a copy' -> move the copy ('Copy of full_data.zip') into this folder "
            "(README, section 8.3).\nThe links come by e-mail after a free registration at "
            "http://got-10k.aitestunion.com/downloads ; download links or archive paths can also be passed as "
            "got10k_urls.")
    archives = _without_duplicates(archives)
    tmp = train_dir.parent / TMP_NAME
    if tmp.exists():
        shutil.rmtree(tmp)
    _check_space(tmp, sum(_uncompressed_size(a, GOT10K_MEMBERS) for a in archives), "GOT-10k")
    try:
        for a in archives:
            extract(a, tmp / "extracted", GOT10K_MEMBERS)
        # Archives inside archives (e.g. full_data.zip -> .../GOT-10k_Train_split_01.zip, ...): extract them one
        # at a time and delete each (temporary) copy right away to limit the disk usage.
        nested, sequences = _walk(tmp / "extracted")
        while nested:
            for a in nested:
                _check_space(tmp, _uncompressed_size(a, GOT10K_MEMBERS), a.name)
                extract(a, a.parent, GOT10K_MEMBERS)
                a.unlink()
            nested, sequences = _walk(tmp / "extracted")
        if len(sequences) != GOT10K_TRAIN_SEQUENCES:
            raise RuntimeError(f"The archives in {archive_dir} contain {len(sequences)} GOT-10k train sequences; "
                               f"all {GOT10K_TRAIN_SEQUENCES} are needed (all train archives).")
        out = tmp / "train"
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
        os.replace(out, train_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if delete_archives:  # only the copies in the archive folder, never archives given by path
        for a in archives:
            if archive_dir.resolve() in a.resolve().parents:
                a.unlink()
    _log(f"GOT-10k ready: {train_dir} ({len(names)} sequences)")
    return train_dir


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
                         "(see README, 'Training').")
    out = {}
    if "coco" in datasets:
        out["coco"] = str(prepare_coco(train_data / "coco", archive_dir / "coco", delete_archives))
    if "got10k" in datasets:
        out["got10k"] = str(prepare_got10k(train_data / "got10k" / "train", archive_dir / "got10k", got10k_urls,
                                           delete_archives))
    return out
