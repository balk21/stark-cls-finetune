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

from stark_ft import archives as arc
from stark_ft import download as dl
from stark_ft import got10k
from stark_ft.train import data as td

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
        saved = dict(got10k.SPLITS), td.COCO_TRAIN_IMAGES, dict(td.COCO_URLS)
        got10k.SPLITS.update(train=("Train", len(NAMES)), val=("Val", 2), test=("Test", 2))
        td.COCO_TRAIN_IMAGES = 7
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fn(Path(tmp))
        finally:
            got10k.SPLITS.update(saved[0])
            td.COCO_TRAIN_IMAGES = saved[1]
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
    assert not (train.parent / f"{arc.TMP_NAME}_train").exists() and (arch / "full_data.zip").is_file()
    mtime = (train / "list.txt").stat().st_mtime
    td.prepare(["got10k"], tmp / "data", tmp / "archives")  # ready: nothing is touched
    assert (train / "list.txt").stat().st_mtime == mtime


@_with_small_counts
def test_got10k_same_archive_twice_is_extracted_once(tmp):
    # e.g. "Copy of full_data.zip" (own copy) and "full_data.zip" (a shortcut to the shared file) in the same folder
    arch = tmp / "archives" / "got10k"
    arch.mkdir(parents=True)
    with zipfile.ZipFile(arch / "Copy of full_data.zip", "w") as z:
        z.writestr("full_data/train_data/GOT-10k_Train_split_01.zip", _split_zip(NAMES))
    (arch / "full_data.zip").write_bytes((arch / "Copy of full_data.zip").read_bytes())
    extracted, original = [], arc.extract
    arc.extract = lambda a, *args, **kw: (extracted.append(a.name), original(a, *args, **kw))[1]
    try:
        train = Path(td.prepare(["got10k"], tmp / "data", tmp / "archives")["got10k"])
    finally:
        arc.extract = original
    assert extracted[0] == "Copy of full_data.zip" and "full_data.zip" not in extracted
    assert sorted(p.name for p in train.iterdir() if p.is_dir()) == NAMES


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
    assert not (tmp / "data" / "got10k" / f"{arc.TMP_NAME}_train").exists()
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
        first = dl.download(f"{base}/train2017.zip", tmp / "dl")
    finally:
        server.shutdown()
    assert dl.download(f"{base}/train2017.zip", tmp / "dl") == first


@_with_small_counts
def test_got10k_full_data_layout_and_path_sources(tmp):
    # Layout of the official full_data.zip: train/ (videos + list.txt), val/, test/ - only train/ is extracted
    src = tmp / "elsewhere" / "full_data.zip"  # e.g. a shortcut on the mounted Google Drive
    src.parent.mkdir()
    with zipfile.ZipFile(src, "w") as z:
        for n in NAMES:
            z.writestr(f"train/{n}/groundtruth.txt", "1,2,3,4\n")
            z.writestr(f"train/{n}/00000001.jpg", b"x" * 100)
        z.writestr("train/list.txt", "\n".join(NAMES))
        z.writestr("val/GOT-10k_Val_000001/00000001.jpg", b"v" * 5000)
        z.writestr("test/GOT-10k_Test_000001/00000001.jpg", b"t" * 5000)
    assert arc.uncompressed_size(src) - arc.uncompressed_size(src, got10k.members("train")) == 10000  # val + test
    for no_unzip in (False, True):  # with the unzip program and with the zipfile fallback
        which = arc.shutil.which
        if no_unzip:
            arc.shutil.which = lambda name: None
        try:
            data = tmp / f"data{int(no_unzip)}"
            train = Path(td.prepare(["got10k"], data, tmp / "archives", [str(src)], delete_archives=True)["got10k"])
        finally:
            arc.shutil.which = which
        assert sorted(p.name for p in train.iterdir() if p.is_dir()) == NAMES
        assert (train / "list.txt").read_text() == "\n".join(NAMES)
        assert src.is_file()  # archives given by path are never deleted
    try:
        td.prepare(["got10k"], tmp / "data2", tmp / "archives", [str(tmp / "missing.zip")])
        raise AssertionError("missing source accepted")
    except FileNotFoundError:
        pass


def test_google_drive_links_and_web_pages():
    file_id = "1b75MBq7MbDQUc682IoECIekoRim_Ydk1"
    direct = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    for url in (f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
                f"https://drive.google.com/open?id={file_id}", f"https://drive.google.com/uc?export=download&id={file_id}"):
        assert dl.direct_url(url) == direct
    assert dl.direct_url("http://images.cocodataset.org/zips/train2017.zip").endswith("train2017.zip")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "page.html").write_text("<html>quota exceeded</html>")
        handler = functools.partial(_QuietHandler, directory=str(tmp))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            dl.download(f"http://127.0.0.1:{server.server_address[1]}/page.html", tmp / "dl")
            raise AssertionError("web page accepted as a download")
        except RuntimeError as e:
            assert "web page" in str(e)
        finally:
            server.shutdown()
        assert not (tmp / "dl" / dl.URL_MAP).exists() and not list((tmp / "dl").glob("page*"))


@_with_small_counts
def test_got10k_val_and_test_splits(tmp):
    # The same full_data.zip also gives the val and test splits; only their members are extracted
    arch = tmp / "archives" / "got10k"
    arch.mkdir(parents=True)
    val, test = ["GOT-10k_Val_000001", "GOT-10k_Val_000002"], ["GOT-10k_Test_000001", "GOT-10k_Test_000002"]
    with zipfile.ZipFile(arch / "full_data.zip", "w") as z:
        for n in NAMES:
            z.writestr(f"train/{n}/groundtruth.txt", "1,2,3,4\n")
            z.writestr(f"train/{n}/00000001.jpg", b"x" * 100)
        for n in val:
            z.writestr(f"val/{n}/groundtruth.txt", "1,2,3,4\n5,6,7,8\n")
            z.writestr(f"val/{n}/cover.label", "8\n0\n")
        for n in test:
            z.writestr(f"test/{n}/groundtruth.txt", "1,2,3,4\n")
        z.writestr("val/list.txt", "\n".join(val))
    root = tmp / "data" / "got10k"
    val_dir = got10k.prepare_split("val", root, arch)
    assert sorted(p.name for p in val_dir.iterdir() if p.is_dir()) == val and (val_dir / "list.txt").is_file()
    assert not (root / "train").exists() and not (root / "test").exists()  # only the requested split
    test_dir = got10k.prepare_split("test", root, arch)
    assert (test_dir / "list.txt").read_text() == "\n".join(test)  # generated: the archive has no test list.txt
    assert got10k.all_sequences("val") == val


def test_got10k_read_sequence():
    import cv2
    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        seq = Path(tmp) / "GOT-10k_Val_000001"
        seq.mkdir()
        for i in range(1, 4):
            cv2.imwrite(str(seq / f"{i:08d}.jpg"), np.zeros((20, 30, 3), np.uint8))
        (seq / "groundtruth.txt").write_text("1,2,3,4\n2,3,4,5\n3,4,5,6\n")
        (seq / "cover.label").write_text("8\n0\n3\n")
        images, boxes, size, init = got10k.read_sequence(Path(tmp), seq.name)
        assert len(images) == 3 and size == (30, 20) and init == [1, 2, 3, 4]
        assert boxes == [[1, 2, 3, 4], None, [3, 4, 5, 6]]  # cover 0: not visible (not evaluated)
        (seq / "cover.label").unlink()
        (seq / "groundtruth.txt").write_text("1,2,3,4\n")  # test split: first frame only
        assert got10k.read_sequence(Path(tmp), seq.name)[1] == [[1, 2, 3, 4], None, None]


if __name__ == "__main__":  # without pytest: python -m tests.test_train_data
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
