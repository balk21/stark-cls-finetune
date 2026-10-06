"""
Downloads with resuming, shared by the training data (COCO, GOT-10k) and the test data (VOT sequences).

A finished download is remembered by its URL (<dst_dir>/.downloads.json), so links that expire (e.g. e-mailed ones)
are not needed again. Google Drive share links are converted to direct downloads; a web page instead of a file
(e.g. Google Drive's "Quota exceeded") raises an error that says what to do.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

URL_MAP = ".downloads.json"  # url -> file name of finished downloads
_DRIVE_ID = re.compile(r"(?:drive\.google\.com/(?:file/d/|open\?(?:[^#]*&)?id=|uc\?(?:[^#]*&)?id=)"
                       r"|drive\.usercontent\.google\.com/download\?(?:[^#]*&)?id=)([\w-]{20,})")


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _gb(n):
    return f"{n / 1e9:.1f} GB"


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
              "Colab <colab_drive>/train_archives/got10k/). Your own copy has no such limit. See docs/colab.md.")


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
