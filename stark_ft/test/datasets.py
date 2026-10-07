"""
Test datasets (the `dataset` test parameter):

  "votlt2020"     VOT-LT2020 (50 sequences), run with vot-toolkit; sequences are downloaded on demand (vot_data.py)
  "got10k_val"    GOT-10k validation split (180 sequences, ground truth available)
  "got10k_test"   GOT-10k test split (180 sequences; first-frame box only: the results are scored by the GOT-10k server)
  "got10k_train"  GOT-10k train split (9335 sequences; note: the *_got10k_only weights were trained on it)
GOT-10k splits are run with the GOT-10k protocol (one pass, no restarts; stark_ft/test/direct.py) and extracted from
the GOT-10k archives on first use (stark_ft/got10k.py).
"""
from pathlib import Path

from stark_ft import got10k
from stark_ft.paths import Paths
from stark_ft.test import vot_data

VOT = "votlt2020"
GOT10K = {"got10k_train": "train", "got10k_val": "val", "got10k_test": "test"}
DATASETS = (VOT, *GOT10K)


def is_got10k(dataset: str) -> bool:
    return dataset in GOT10K


def has_ground_truth(dataset: str) -> bool:
    return dataset != "got10k_test"


def root(dataset: str, paths: Paths) -> Path:
    """The folder with the dataset's sequences."""
    return paths.train_data / "got10k" / GOT10K[dataset] if is_got10k(dataset) else paths.dataset


def resolve(dataset: str, sequences, paths: Paths) -> list:
    """'all' -> every sequence of the dataset; a list is checked against the dataset's sequence names."""
    if not is_got10k(dataset):
        return vot_data.resolve(sequences, paths.dataset)
    names = got10k.all_sequences(GOT10K[dataset])
    if sequences == "all":
        return names
    unknown = [s for s in sequences if s not in set(names)]
    if unknown:
        raise ValueError(f"Not in GOT-10k {GOT10K[dataset]}: {unknown}\n"
                         f"Its sequences are {names[0]} ... {names[-1]}")
    return list(sequences)


def to_fetch(dataset: str, sequences, paths: Paths) -> str:
    """What has to be downloaded / extracted before a run ('' if nothing)."""
    if not is_got10k(dataset):
        todo = vot_data.missing(sequences, paths.dataset)
        return f"{len(todo)} sequence(s) not on disk yet ({vot_data.size_text(todo)})" if todo else ""
    if got10k.ready(root(dataset, paths)):
        return ""
    return f"GOT-10k {GOT10K[dataset]} split: extracted from the GOT-10k archives in {paths.archives / 'got10k'}"


def ensure(dataset: str, sequences, paths: Paths):
    """Downloads (VOT) or extracts (GOT-10k) what is missing."""
    if is_got10k(dataset):
        got10k.prepare_split(GOT10K[dataset], paths.train_data / "got10k", paths.archives / "got10k")
    else:
        vot_data.ensure(sequences, paths.dataset, paths.dataset_cache)


def ground_truth(dataset: str, dataset_dir: Path, seq: str):
    """(boxes per frame with None where the target is not visible, image size (w, h) or None)."""
    if is_got10k(dataset):
        _, boxes, size, _ = got10k.read_sequence(dataset_dir, seq)
        return boxes, size
    from stark_ft.test.evaluation import read_groundtruth
    return read_groundtruth(Path(dataset_dir) / seq / "groundtruth.txt"), None
