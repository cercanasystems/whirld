"""Single source of truth for the Whirld version string.

Kept in its own module so it can be imported without triggering any heavy
imports (numpy, rasterio, etc.) — see ``whirld.__init__`` for the lazy public API.
"""

from __future__ import annotations

__version__ = "0.1.0"
