"""RemoteCLIP backend — zero-shot text-driven classification via open_clip.

RemoteCLIP is an OpenAI-CLIP ViT-B/32 fine-tuned for remote sensing. Given image
chips and a free-text query, it scores how well each chip matches the query
(cosine similarity between the CLIP image and text embeddings).

Requires the ``remoteclip`` extra (``open_clip_torch``). Unlike Clay, no vendoring
is needed — ``open_clip_torch`` installs cleanly on Python 3.13.

**Preprocessing note.** RemoteCLIP expects RGB pixels scaled to 0..1 then
CLIP-standardized at 224x224. The band contract already produces exactly that
(scale + CLIP mean/std + 224 chip), so chips arrive model-ready and this backend
does no further image preprocessing.

**Scoring.** ``logit_scale``-scaled cosine + softmax over the prompts → per-chip,
per-query **probabilities** in ``[0, 1]`` (RemoteCLIP's own scoring). Multiple
queries form a zero-shot classification over those classes; a single query is
softmaxed against a neutral ``"background"`` prompt for a calibrated match score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config
from ..core.fetch import Manifest
from ..core.registry import ModelEntry
from ..errors import ModelNotInstalledError, WhirldError
from ..logging_setup import get_logger
from .base import InferenceContext, ModelBackend, detect_device

_log = get_logger("models.remoteclip")

_DEFAULT_BATCH = 16
# Neutral contrast prompt used when a single query is given, so the softmax yields
# a calibrated match probability (query vs. background) rather than a constant 1.0.
_BACKGROUND_PROMPTS = ["background"]


def _import_open_clip():
    """Lazily import open_clip, raising an actionable error if the extra is absent.

    Returns:
        The imported ``open_clip`` module.

    Raises:
        WhirldError: ``open_clip`` is not installed.
    """
    try:
        import open_clip  # noqa: PLC0415

        return open_clip
    except ImportError as exc:
        raise WhirldError(
            "RemoteCLIP requires the optional 'remoteclip' extra (open_clip_torch).\n"
            "       Install it:  pip install 'whirld[remoteclip]'"
        ) from exc


class RemoteCLIPBackend(ModelBackend):
    """open_clip-backed RemoteCLIP classification backend.

    Args:
        name: Model identifier.
        version: Model version.
        device: Resolved inference device string.
        arch: open_clip architecture name (e.g. ``ViT-B-32``).
        ckpt_path: Path to the downloaded ``.pt`` (``None`` if a model is injected).
        model: A prebuilt model with ``encode_image``/``encode_text``/``logit_scale``
            (used by tests instead of loading weights).
        tokenizer: A callable turning a list of strings into a token tensor
            (used by tests).
        batch_size: Inference batch size over chips.
    """

    def __init__(
        self,
        name: str,
        version: str,
        device: str,
        *,
        arch: str,
        ckpt_path: Path | None = None,
        model: object | None = None,
        tokenizer: object | None = None,
        batch_size: int = _DEFAULT_BATCH,
    ) -> None:
        super().__init__(name=name, version=version, device=device)
        self.arch = arch
        self._ckpt_path = ckpt_path
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size

    @classmethod
    def load(
        cls,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None = None,
    ) -> RemoteCLIPBackend:
        """Build the backend and load the real RemoteCLIP weights.

        Args:
            entry: The validated registry entry (provides ``model_name``).
            manifest: The local manifest from ``whirld pull``.
            device: Requested device, or ``None`` for auto-detection.

        Returns:
            A ready :class:`RemoteCLIPBackend` with weights loaded.

        Raises:
            ModelNotInstalledError: The checkpoint file is missing.
            WhirldError: The ``remoteclip`` extra is missing or ``model_name`` is unset.
        """
        resolved = detect_device(device)
        if not entry.model_name:
            raise WhirldError(
                f"Registry entry '{entry.name}' must declare model_name "
                f"(the open_clip architecture)."
            )
        ckpt = config.get_paths().model_dir(entry.name) / manifest.weights_file
        if not ckpt.exists():
            raise ModelNotInstalledError(
                f"Checkpoint for '{entry.name}' is missing at {ckpt}.\n"
                f"       Re-pull it:  whirld pull {entry.name}"
            )
        backend = cls(
            name=manifest.name,
            version=manifest.version,
            device=resolved,
            arch=entry.model_name,
            ckpt_path=ckpt,
        )
        backend._ensure_model()
        return backend

    def _ensure_model(self) -> tuple[object, object]:
        """Build the open_clip model + tokenizer and load weights on first use.

        Returns:
            A tuple ``(model, tokenizer)`` ready for inference.
        """
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        open_clip = _import_open_clip()
        import torch  # noqa: PLC0415

        _log.info(
            "Loading RemoteCLIP (%s) weights on device '%s'.", self.arch, self.device
        )
        model, _, _ = open_clip.create_model_and_transforms(self.arch)
        state_dict = torch.load(self._ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        _log.info(
            "RemoteCLIP weights: %d missing, %d unexpected.",
            len(missing),
            len(unexpected),
        )
        model.to(self.device).eval()
        self._model = model
        self._tokenizer = open_clip.get_tokenizer(self.arch)
        return self._model, self._tokenizer

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """RemoteCLIP is a classification model; embedding is not supported.

        Raises:
            WhirldError: Always — use ``classify`` instead.
        """
        raise WhirldError(
            f"Model '{self.name}' does not produce embeddings; use 'classify'."
        )

    def classify(
        self,
        chips: np.ndarray,
        queries: list[str],
        context: InferenceContext | None = None,
    ) -> np.ndarray:
        """Score each chip against ``queries`` as zero-shot probabilities.

        Uses RemoteCLIP's own scoring: L2-normalized image/text features,
        ``logit_scale``-scaled cosine, then softmax over the prompts. With a single
        query, a neutral ``"background"`` prompt is added so the softmax yields a
        calibrated 0–1 match probability rather than a constant 1.0. The returned
        columns are the user queries only (the background column is dropped).

        Args:
            chips: Array ``(n_chips, 3, height, width)``, float32, already
                CLIP-normalized by the band contract.
            queries: One or more free-text queries (e.g. ``["solar farm"]``).
            context: Unused (RemoteCLIP needs no spectral metadata).

        Returns:
            Per-chip, per-query probabilities, shape ``(n_chips, len(queries))``,
            float32, each in ``[0, 1]``.

        Raises:
            ValueError: ``chips`` is not 4-D.
            WhirldError: ``queries`` is empty or all-blank.
        """
        if chips.ndim != 4:
            raise ValueError(
                f"Expected chips of shape (n, bands, h, w), got {chips.shape}."
            )
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            raise WhirldError("classify requires at least one non-empty --query.")
        if chips.shape[0] == 0:
            return np.empty((0, len(queries)), dtype=np.float32)

        # Single query needs a contrasting prompt for a meaningful softmax.
        prompts = queries + (_BACKGROUND_PROMPTS if len(queries) == 1 else [])
        n_q = len(queries)

        import torch  # noqa: PLC0415

        model, tokenizer = self._ensure_model()
        tokens = tokenizer(prompts).to(self.device)
        scores: list[np.ndarray] = []
        with torch.no_grad():
            scale = model.logit_scale.exp()
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            for start in range(0, chips.shape[0], self._batch_size):
                batch = chips[start : start + self._batch_size]
                pixels = (
                    torch.from_numpy(np.ascontiguousarray(batch))
                    .float()
                    .to(self.device)
                )
                image_features = model.encode_image(pixels)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                logits = scale * (image_features @ text_features.T)
                probs = logits.softmax(dim=-1)[:, :n_q]
                scores.append(probs.cpu().numpy())
        return np.concatenate(scores, axis=0).astype(np.float32)
