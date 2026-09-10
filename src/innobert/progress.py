"""Progress-display helpers that remain silent in non-interactive jobs."""

import sys


def progress_iter(iterable, *, total=None, description=None, progress="auto"):
    """Wrap an iterable in tqdm when progress is enabled."""
    enabled = resolve_progress(progress)
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise ImportError("Progress display requires tqdm. Reinstall InnoBERT or use progress=False.") from exc
    return tqdm(iterable, total=total, desc=description, dynamic_ncols=True, leave=True)


def resolve_progress(progress):
    if isinstance(progress, bool):
        return progress
    if progress != "auto":
        raise ValueError("progress must be 'auto', True, or False.")
    if _in_notebook():
        return True
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _in_notebook():
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except (NameError, AttributeError):
        return False
    return shell == "ZMQInteractiveShell"
