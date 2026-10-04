"""Training data preparation with small fake archives (no network access: COCO is served by a local HTTP server)."""
import functools
import http.server
import io
import json
import os
import tempfile
import threading
import zipfile
from pathlib import Path

from stark_ft import train_data as td

NAMES = [f"GOT-10k_Train_{i:06d}" for i in range(1, 6)]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _split_zip(names) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n in names:
            z.writestr(f"{n}/groundtruth.txt", "1,2,3,4\n")
            z.writestr(f"{n}/00000001.jpg", b"x" * 100)
    return buf.getvalue()


def _with_small_counts(fn):
    @functools.wraps(fn)
    def wrapper():
        saved = td.GOT10K_TRAIN_SEQUENCES, td.COCO_TRAIN_IMAGES, dict(td.COCO_URLS)
        td.GOT10K_TRAIN_SEQUENCES, td.COCO_TRAIN_IMAGES = len(NAMES), 7
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fn(Path(tmp))
        finally:
            td.GOT10K_TRAIN_SEQUENCES, td.COCO_TRAIN_IMAGES = saved[:2]
            td.COCO_URLS.clear()
            td.COCO_URLS.update(saved[2])
    return wrapper


@_with_small_counts
def test_got10k_nested_archives(tmp):
    arch = tmp / "archives" / "got10k"
    arch.mkdir(parents=True)
    with zipfile.ZipFile(arch / "full_data.zip", "w") as z:  # archives inside an archive + the official list.txt
        z.writestr("full_data/train_data/GOT-10k_Train_split_01.zip", _split_zip(NAMES[:3]))
        z.writestr("full_data/train_data/GOT-10k_Train_split_02.zip", _split_zip(NAMES[3:]))
        z.writestr("full_data/train_data/list.txt", "\n".join(NAMES))
    train = Path(td.prepare(["got10k"], tmp / "data", tmp / "archives")["got10k"])
    assert sorted(p.name for p in train.iterdir() if p.is_dir()) == NAMES
    assert (train / "list.txt").read_text() == "\n".join(NAMES)
    assert not (train.parent / td.TMP_NAME).exists() and (arch / "full_data.zip").is_file()
    mtime = (train / "list.txt").stat().st_mtime
    td.prepare(["got10k"], tmp / "data", tmp / "archives")  # ready: nothing is touched
    assert (train / "list.txt").stat().st_mtime == mtime


@_with_small_counts
def test_got10k_split_archives_and_errors(tmp):
    arch = tmp / "archives" / "got10k"
    arch.mkdir(parents=True)
    (arch / "GOT-10k_Train_split_01.zip").write_bytes(_split_zip(NAMES[:2]))
    try:  # incomplete: error, no half-prepared folder
        td.prepare(["got10k"], tmp / "data", tmp / "archives")
        raise AssertionError("incomplete archives accepted")
    except RuntimeError:
        pass
    assert not (tmp / "data" / "got10k" / "train").exists()
    assert not (tmp / "data" / "got10k" / td.TMP_NAME).exists()
    (arch / "GOT-10k_Train_split_02.zip").write_bytes(_split_zip(NAMES[2:]))
    train = Path(td.prepare(["got10k"], tmp / "data", tmp / "archives")["got10k"])
    assert (train / "list.txt").read_text() == "\n".join(NAMES)  # generated: same format as the official one
    try:
        td.prepare(["got10k"], tmp / "data2", tmp / "empty")
        raise AssertionError("missing archives accepted")
    except FileNotFoundError as e:
        assert "registration" in str(e)


@_with_small_counts
def test_coco_download_and_extract(tmp):
    server_dir = tmp / "server"
    server_dir.mkdir()
    with zipfile.ZipFile(server_dir / "train2017.zip", "w") as z:
        for i in range(7):
            z.writestr(f"train2017/{i:012d}.jpg", os.urandom(1000))
    with zipfile.ZipFile(server_dir / "annotations_trainval2017.zip", "w") as z:
        z.writestr("annotations/instances_train2017.json", json.dumps({"images": []}))
    handler = functools.partial(_QuietHandler, directory=str(server_dir))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        td.COCO_URLS.update({k: f"{base}/{k}" for k in td.COCO_URLS})
        archives = tmp / "archives"
        (archives / "coco").mkdir(parents=True)
        # an interrupted download (this server ignores Range requests, so the download starts again)
        (archives / "coco" / "train2017.zip.part").write_bytes(b"partial")
        coco = Path(td.prepare(["coco"], tmp / "data", archives)["coco"])
        assert len(list((coco / "images" / "train2017").iterdir())) == 7
        assert (coco / "annotations" / "instances_train2017.json").is_file()
        assert (archives / "coco" / "train2017.zip").read_bytes() == (server_dir / "train2017.zip").read_bytes()
        # a finished download is found again by its URL without contacting the server (e-mailed links expire)
        first = td.download(f"{base}/train2017.zip", tmp / "dl")
    finally:
        server.shutdown()
    assert td.download(f"{base}/train2017.zip", tmp / "dl") == first


if __name__ == "__main__":  # without pytest: python -m tests.test_train_data
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
