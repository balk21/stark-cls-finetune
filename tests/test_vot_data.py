"""On-demand VOT sequence download with small fake sequences (no network access: a local HTTP server stands in for
data.votchallenge.net, with the same description.json layout)."""
import contextlib
import functools
import hashlib
import http.server
import io
import json
import tempfile
import threading
import zipfile
from pathlib import Path

from stark_ft.paths import Paths
from stark_ft.test import vot_data
from stark_ft.test.config import ExperimentConfig
from stark_ft.test.runner import ExperimentExistsError, prepare_experiment, resolve_sequences

NAMES = ["seqa", "seqb"]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def _entry(server_dir: Path, url: str, data: bytes) -> dict:
    (server_dir / url).parent.mkdir(parents=True, exist_ok=True)
    (server_dir / url).write_bytes(data)
    return {"url": url, "checksum": hashlib.sha1(data).hexdigest(), "compressed": len(data)}


@contextlib.contextmanager
def _server(tmp: Path, corrupt=()):
    """Serves vot2019/longterm/description.json with the sequences in NAMES (uid-named color archives two folders up,
    as on the official server). Sequences in `corrupt` get a wrong checksum."""
    root = tmp / "server"
    longterm = root / "vot2019" / "longterm"
    sequences = []
    for name in NAMES:
        ann = _entry(longterm, f"{name}.zip", _zip({"groundtruth.txt": "1,2,3,4\nnan,nan,nan,nan\n"}))
        color = _entry(longterm, f"../../sequences/{name}_uid.zip",
                       _zip({"00000001.jpg": b"a" * 50, "00000002.jpg": b"b" * 50}))
        color["pattern"] = "%08d.jpg"
        if name in corrupt:
            color["checksum"] = "0" * 40
        sequences.append({"name": name, "fps": 30, "annotations": ann, "channels": {"color": color}})
    (longterm / "description.json").write_text(json.dumps({"name": "fake", "sequences": sequences}))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                             functools.partial(_QuietHandler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    saved = vot_data.DESCRIPTION_URL, dict(vot_data.SEQUENCES)
    vot_data.DESCRIPTION_URL = f"http://127.0.0.1:{server.server_address[1]}/vot2019/longterm/description.json"
    vot_data.SEQUENCES.clear()
    vot_data.SEQUENCES.update({n: 1 for n in NAMES})
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        vot_data.DESCRIPTION_URL = saved[0]
        vot_data.SEQUENCES.clear()
        vot_data.SEQUENCES.update(saved[1])


def _paths(tmp: Path, cache=True) -> Paths:
    return Paths(checkpoints=tmp / "ckpt", dataset=tmp / "seq", outputs=tmp / "out", train_data=tmp / "train_data",
                 train_outputs=tmp / "train_out", dataset_cache=(tmp / "cache") if cache else None)


def test_resolve_names():
    with tempfile.TemporaryDirectory() as tmp:
        assert vot_data.resolve("all", Path(tmp)) == sorted(vot_data.SEQUENCES) and len(vot_data.SEQUENCES) == 50
        assert vot_data.resolve(["bull", "ballet"], Path(tmp)) == ["bull", "ballet"]
        try:
            vot_data.resolve(["bul"], Path(tmp))
            raise AssertionError("unknown sequence accepted")
        except ValueError as e:
            assert "bul" in str(e) and "ballet" in str(e)
        own = Path(tmp) / "my_sequence"  # a sequence of an own dataset is accepted when it is on disk
        own.mkdir()
        (own / "sequence").write_text("name=my_sequence\n")
        (own / "groundtruth.txt").write_text("1,2,3,4\n")
        assert vot_data.resolve(["my_sequence"], Path(tmp)) == ["my_sequence"]


def test_download_only_requested_and_cache():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with _server(tmp):
            assert vot_data.ensure(["seqa"], tmp / "seq", tmp / "cache") == ["seqa"]
            assert vot_data.ensure(["seqa"], tmp / "seq", tmp / "cache") == []  # present: nothing fetched
        seq = tmp / "seq" / "seqa"
        assert sorted(p.name for p in (tmp / "seq").iterdir()) == ["list.txt", "seqa"]  # seqb not downloaded
        assert sorted(p.name for p in (seq / "color").iterdir()) == ["00000001.jpg", "00000002.jpg"]
        # the `sequence` file exactly as vot-toolkit writes it (key-sorted, csv line endings)
        assert (seq / "sequence").read_bytes() == (b"channels.color=color/%08d.jpg\r\nformat=default\r\nfps=30\r\n"
                                                   b"name=seqa\r\n")
        assert (seq / "groundtruth.txt").read_text().startswith("1,2,3,4")
        assert (tmp / "seq" / "list.txt").read_text() == "seqa\n"
        assert [p.name for p in (tmp / "cache").iterdir()] == ["seqa.tar"]
        with _server(tmp):  # a new machine / Colab session: restored from the cache, without the server
            vot_data.DESCRIPTION_URL = "http://127.0.0.1:9/unreachable/description.json"
            assert vot_data.ensure(["seqa"], tmp / "seq2", tmp / "cache") == ["seqa"]
        assert (tmp / "seq2" / "seqa" / "sequence").read_bytes() == (seq / "sequence").read_bytes()
        assert not (tmp / "seq2" / vot_data.TMP_DIR).exists()


def test_checksum_mismatch_leaves_no_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with _server(tmp, corrupt=["seqb"]):
            try:
                vot_data.ensure(["seqa", "seqb"], tmp / "seq")
                raise AssertionError("corrupt archive accepted")
            except RuntimeError as e:
                assert "Checksum" in str(e)
        assert vot_data.missing(["seqa", "seqb"], tmp / "seq") == ["seqb"]  # seqa is complete, seqb absent
        assert not (tmp / "seq" / "seqb").exists()


def test_experiment_downloads_its_sequences_after_the_checks():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = _paths(tmp)
        cfg = ExperimentConfig(sequences=["seqb"], ft_mode="none", name="exp")
        ckpt = cfg.checkpoint_path(paths)
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"weights")
        with _server(tmp):
            assert resolve_sequences(cfg, paths, fetch=False) == ["seqb"]  # validation only: nothing downloaded
            assert not (tmp / "seq").exists()
            # an existing experiment with other parameters: error before anything is downloaded
            (paths.outputs / "exp").mkdir(parents=True)
            other = ExperimentConfig(sequences=["seqb"], ft_mode="init", name="exp")
            (paths.outputs / "exp" / "experiment.json").write_text(json.dumps({"config": other.to_dict()}))
            try:
                prepare_experiment(cfg, paths)
                raise AssertionError("different parameters accepted")
            except ExperimentExistsError:
                pass
            assert not (tmp / "seq").exists()
            out = prepare_experiment(cfg, paths, overwrite=True)
        assert vot_data.missing(["seqa", "seqb"], paths.dataset) == ["seqa"]
        listed = (out / "vot_workspace" / "sequences" / "list.txt").read_text().split()
        assert listed == [str((paths.dataset / "seqb").resolve())]
        assert (tmp / "cache" / "seqb.tar").is_file()


if __name__ == "__main__":  # without pytest: python -m tests.test_vot_data
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("PASS", _name)
