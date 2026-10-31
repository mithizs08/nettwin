"""Split guard on data paths (audit Phase 5, R5): every read under a dataset's `observable/` or `truth/` tree is checked
against the splits the calling script is allowed to read, from the path itself rather than from argv."""
from pathlib import Path

TREES = ("observable", "truth")


def split_of(path):
    """The split a dataset path belongs to (the folder right after `observable` or `truth`), or None."""
    parts = Path(path).resolve().parts
    for i, p in enumerate(parts[:-1]):
        if p in TREES:
            return parts[i + 1]
    return None


def guard(path, allowed):
    """Return `path` if it is inside a dataset split listed in `allowed`; raise PermissionError otherwise."""
    s = split_of(path)
    if s not in allowed:
        raise PermissionError(f"{path}: split {s!r} is not one of {sorted(allowed)}")
    return path
