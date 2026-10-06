"""
Test data: the VOT-LT2020 sequences (downloaded on demand by stark_ft/test/vot_data.py).
"""
from pathlib import Path

from stark_ft.paths import get_paths, list_sequences
from stark_ft.test import vot_data


def download_dataset(target=None, sequences="all"):
    """Downloads VOT-LT2020 (LTB50) sequences that are not there yet: "all" (50 sequences, 17.6 GB) or a list of
    names. Not needed before experiments: every experiment downloads the sequences it uses."""
    paths = get_paths()
    target = Path(target) if target else paths.dataset
    names = vot_data.resolve(sequences, target)
    fetched = vot_data.ensure(names, target, paths.dataset_cache)
    print(f"{len(names) - len(fetched)} of {len(names)} sequence(s) were already present, {len(fetched)} fetched. "
          f"Dataset: {target} ({len(list_sequences(target))} sequences)")
    return target
