"""Vendored Clay v1.5 encoder (Apache-2.0). See ``PROVENANCE.md`` and ``NOTICE``.

Only the encoder needed to produce embeddings is vendored, not the full training
model. Exposes :class:`Encoder` and the :func:`clay_v15_large_encoder` factory.
"""

from .encoder import Encoder, clay_v15_large_encoder

__all__ = ["Encoder", "clay_v15_large_encoder"]
