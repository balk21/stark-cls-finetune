"""
Loading STARK checkpoints.

Besides the weights ('net'), the official checkpoints also contain training metadata (settings, stats ...)
pickled with `lib.train.admin.*` classes. These classes are not needed for inference; to avoid shipping the
training code, they are replaced with empty placeholders while loading.
"""
import pickle
import types

import torch

_STUB_PREFIXES = ("lib.train", "ltr.")


class _Placeholder:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(_STUB_PREFIXES):
            return type(name, (_Placeholder,), {"__module__": module})
        return super().find_class(module, name)


_pickle_module = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="pickle")


def load_network_weights(path):
    """Returns only the network weights (state_dict) from a checkpoint file."""
    try:
        ckpt = torch.load(path, map_location="cpu", pickle_module=_pickle_module, weights_only=False)
    except TypeError:  # older PyTorch versions without the weights_only argument
        ckpt = torch.load(path, map_location="cpu", pickle_module=_pickle_module)
    return ckpt["net"] if isinstance(ckpt, dict) and "net" in ckpt else ckpt
