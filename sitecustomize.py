"""Compatibility shim.

Older versions of this project monkey‑patched the bot loop here. That logic
has diverged and was causing import-time failures. The live bot now handles
control semantics directly, so this file is intentionally a no-op.
"""

from __future__ import annotations

try:
    import bot  # noqa: F401
except Exception as exc:  # pragma: no cover
    print(f"[SITECUSTOMIZE] import skipped: {exc}", flush=True)
else:
    print("[SITECUSTOMIZE] compatibility shim loaded", flush=True)
