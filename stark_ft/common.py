"""Helpers shared by training and testing: parameter type coercion and code hashes."""
import hashlib
from typing import List, Union

from stark_ft.paths import REPO_ROOT


def coerce(name, value, type_):
    """Converts a parameter value to its declared type: "3" -> 3, "1e-5" -> 1e-5 (YAML 1.1 reads it as a string),
    "false" -> False, ["1", "2"] -> [1.0, 2.0] for List[float], "none" -> None for Optional[...]."""
    origin, args = getattr(type_, "__origin__", None), getattr(type_, "__args__", ())
    if origin is Union:
        if type(None) in args and (value is None or (isinstance(value, str)
                                                     and value.strip().lower() in ("none", "null", ""))):
            return None
        inner = [a for a in args if a is not type(None)]
        return coerce(name, value, inner[0]) if len(inner) == 1 else value
    if origin in (list, List):
        return [coerce(name, v, args[0]) for v in value] if isinstance(value, (list, tuple)) else value
    if not isinstance(value, str):
        if type_ is float and isinstance(value, int) and not isinstance(value, bool):
            return float(value)
        return value
    try:
        if type_ is float:
            return float(value)
        if type_ is int:
            return int(value)
        if type_ is bool:
            if value.strip().lower() in ("true", "yes", "1"):
                return True
            if value.strip().lower() in ("false", "no", "0"):
                return False
            raise ValueError
    except ValueError:
        raise ValueError(f"Parameter {name}={value!r} cannot be converted to {type_.__name__}") from None
    return value


def code_hash(entries, exclude=()) -> str:
    """Hash of the .py / .yaml files under `entries` (paths relative to the repository root), independent of git and
    of documentation changes. Runs are not resumed with a different code version."""
    h = hashlib.sha1()
    for entry in entries:
        root = REPO_ROOT / entry
        for f in (sorted(root.rglob("*")) if root.is_dir() else [root]):
            rel = f.relative_to(REPO_ROOT).as_posix()
            if any(rel == e or rel.startswith(e + "/") for e in exclude):
                continue
            if f.is_file() and f.suffix in (".py", ".yaml"):
                h.update(str(f.relative_to(REPO_ROOT)).encode())
                h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]
