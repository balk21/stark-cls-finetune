"""
Archive helpers shared by the training data (COCO, GOT-10k train) and the test data (GOT-10k val / test).
"""
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

from stark_ft.download import QUOTA_HELP, _gb, _log

ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
TMP_NAME = ".stark_prepare_tmp"  # temporary extraction folder, renamed / removed when finished
FREE_MARGIN = 2e9


def is_archive(name: str) -> bool:
    return name.lower().endswith(ARCHIVE_SUFFIXES)


def selected(name: str, members) -> bool:
    return members is None or any(fnmatch.fnmatch(name, pattern) for pattern in members)


def read_error(archive: Path, err) -> RuntimeError:
    msg = f"Cannot read {archive}: {err}"
    if "/drive/" in str(archive):  # Google Drive mount (Colab)
        msg += ("\nIf this is a shortcut to a file shared by someone else, Google Drive probably refuses to serve it "
                "because the file's download quota is used up.\n" + QUOTA_HELP)
    else:
        msg += "\nIf it is an incomplete download, delete it and run the preparation again."
    return RuntimeError(msg)


def uncompressed_size(archive: Path, members=None) -> int:
    if archive.name.lower().endswith(".zip"):
        try:
            with zipfile.ZipFile(archive) as z:
                return sum(i.file_size for i in z.infolist() if selected(i.filename, members))
        except (zipfile.BadZipFile, OSError) as e:
            raise read_error(archive, e) from None
    return archive.stat().st_size * (1 if archive.name.lower().endswith(".tar") else 2)


def without_duplicates(archives):
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
                raise read_error(a, e) from None
            key = (a.stat().st_size, h.hexdigest())
        if key in seen:
            _log(f"Skipping {a.name}: the same archive as {seen[key].name} (one of them can be removed from "
                 f"{a.parent})")
            continue
        seen[key] = a
        kept.append(a)
    return kept


def check_space(where: Path, needed: int, what: str):
    probe = where
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed + FREE_MARGIN:
        raise RuntimeError(f"Not enough disk space for {what}: {_gb(needed)} needed (+{_gb(FREE_MARGIN)} margin), "
                           f"{_gb(free)} free on {probe}. Use a machine / Colab runtime with a larger disk, or fewer "
                           "datasets.")


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
                raise read_error(archive, f"unzip failed (exit code {rc})")
        else:
            with zipfile.ZipFile(archive) as z:
                z.extractall(dst, [n for n in z.namelist() if selected(n, members)])
    else:
        with tarfile.open(archive) as t:
            t.extractall(dst, [m for m in t.getmembers() if selected(m.name, members)])


def find_archives(root: Path, skip_dir=lambda name: False):
    """Archives under root (folders for which skip_dir(name) is true are not entered)."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != TMP_NAME and not skip_dir(d)]
        found += [Path(dirpath) / f for f in filenames if is_archive(f)]
    return sorted(found)
