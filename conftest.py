import os
from pathlib import Path

# Force a non-GUI backend for matplotlib in tests (prevents Tcl/Tk errors).
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure matplotlib has a writable config dir and won't attempt to use system Tcl/Tk paths.
if "MPLCONFIGDIR" not in os.environ:
    cfg = Path.cwd() / ".matplotlib"
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cfg)
