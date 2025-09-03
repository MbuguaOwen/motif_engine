from contextlib import contextmanager
import os
from typing import Iterable, Optional
try:
    from tqdm.auto import tqdm
    from tqdm.contrib.logging import logging_redirect_tqdm
    _TQDM_OK = True
except Exception:
    tqdm = None
    logging_redirect_tqdm = None
    _TQDM_OK = False

def _enabled(yaml_cfg: dict, cli_disable: bool) -> bool:
    if cli_disable:
        return False
    ui = (yaml_cfg or {}).get("ui", {})
    return bool(ui.get("progress", False)) and _TQDM_OK

def _opts(yaml_cfg: dict) -> dict:
    ui = (yaml_cfg or {}).get("ui", {})
    return dict(
        dynamic_ncols=True,
        ascii=bool(ui.get("progress_ascii", True)),
        mininterval=float(ui.get("progress_mininterval", 0.2)),
        leave=True,
    )

def wrap_iter(it: Iterable, total: Optional[int], desc: str, yaml_cfg: dict, cli_disable: bool=False):
    """
    Wrap an iterable with tqdm if enabled; returns original iterable if disabled or tqdm missing.
    """
    if not _enabled(yaml_cfg, cli_disable):
        return it
    return tqdm(it, total=total, desc=desc, **_opts(yaml_cfg))

@contextmanager
def progress_redirect_logs(yaml_cfg: dict, cli_disable: bool=False):
    """
    Context manager that routes logging through tqdm to avoid broken bars.
    """
    if _enabled(yaml_cfg, cli_disable) and logging_redirect_tqdm is not None:
        with logging_redirect_tqdm():
            yield
    else:
        yield

def bar(total: int, desc: str, yaml_cfg: dict, cli_disable: bool=False):
    """
    Create a manual tqdm bar (caller must update .update()).
    """
    if not _enabled(yaml_cfg, cli_disable):
        class _NullBar:
            def update(self, n=1): pass
            def close(self): pass
        return _NullBar()
    return tqdm(total=total, desc=desc, **_opts(yaml_cfg))

