"""RemoteCLIP backend — zero-shot text-driven classification via open_clip.

RemoteCLIP supplies curated CLIP architectures fine-tuned for remote sensing.
Given image chips and a free-text query, it scores how well each chip matches the query
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
softmaxed against a neutral ``"background"`` prompt for a prompt-relative match score.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, Protocol

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor

from .. import config
from ..core.fetch import Manifest
from ..core.registry import ModelEntry
from ..errors import BackendContractError, ModelNotInstalledError, WhirldError
from ..logging_setup import get_logger
from .base import BackendContract, InferenceContext, ModelBackend, detect_device
from .checkpoints import CheckpointReader
from .providers import ProviderCatalog

_log = get_logger("models.remoteclip")

_DEFAULT_BATCH = 16
# Neutral contrast prompt used when a single query is given, so the softmax yields
# a prompt-relative match probability (query vs. background) rather than a constant 1.0.
_BACKGROUND_PROMPTS = ["background"]


class _CLIPModel(Protocol):
    """The inference surface consumed from open_clip, without importing it."""

    logit_scale: Tensor

    def encode_image(self, pixels: Tensor) -> Tensor:
        """Encode a batch of image tensors."""
        ...

    def encode_text(self, tokens: Tensor) -> Tensor:
        """Encode a batch of tokenized prompts."""
        ...


def _import_open_clip() -> ModuleType:
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

    provider_id: ClassVar[str] = "remoteclip"

    def __init__(
        self,
        name: str,
        version: str,
        device: str,
        *,
        arch: str,
        ckpt_path: Path | None = None,
        model: _CLIPModel | None = None,
        tokenizer: Callable[[list[str]], Tensor] | None = None,
        batch_size: int = _DEFAULT_BATCH,
        trusted_checkpoint: bool = False,
    ) -> None:
        super().__init__(
            name=name, version=version, device=device, batch_size=batch_size
        )
        self._contract = BackendContract.create(
            self.name, channels=ProviderCatalog.get(self.provider_id).input_channels
        )
        self.arch = ModelEntry.validate_remoteclip_architecture(arch)
        self._checkpoint = (
            CheckpointReader(ckpt_path, trusted=trusted_checkpoint)
            if ckpt_path is not None
            else None
        )
        self._model = model
        self._tokenizer = tokenizer

    @classmethod
    def load(
        cls,
        entry: ModelEntry,
        manifest: Manifest,
        device: str | None = None,
        *,
        paths: config.WhirldPaths | None = None,
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
        entry.require_operation("classify")
        resolved = detect_device(device)
        if not entry.model_name:
            raise WhirldError(
                f"Registry entry '{entry.name}' must declare model_name "
                f"(the open_clip architecture)."
            )
        ckpt = (paths or config.get_paths()).model_file(
            entry.name, manifest.weights_file
        )
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
            trusted_checkpoint=entry.trust == "first-party",
        )
        backend._ensure_model()
        return backend

    def _ensure_model(self) -> tuple[_CLIPModel, Callable[[list[str]], Tensor]]:
        """Build the open_clip model + tokenizer and load weights on first use.

        Returns:
            A tuple ``(model, tokenizer)`` ready for inference.
        """
        # Validate again at dispatch so direct callers or edited runtime objects
        # cannot route open_clip around the owned checkpoint/source boundary.
        ModelEntry.validate_remoteclip_architecture(self.arch)
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        open_clip = _import_open_clip()

        _log.info(
            "Loading RemoteCLIP (%s) weights on device '%s'.", self.arch, self.device
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.arch, pretrained=None, pretrained_hf=False
        )
        if self._checkpoint is None:
            raise WhirldError("A checkpoint path is required to load RemoteCLIP.")
        state_dict = self._checkpoint.read_state_dict()
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        _log.info(
            "RemoteCLIP weights: %d missing, %d unexpected.",
            len(missing),
            len(unexpected),
        )
        model.to(self.device).eval()
        self._model = model
        tokenizer: Callable[[list[str]], Tensor] = open_clip.get_tokenizer(self.arch)
        self._tokenizer = tokenizer
        return model, tokenizer

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """RemoteCLIP is a classification model; embedding is not supported.

        Raises:
            UnsupportedOperationError: Always; use ``classify`` instead.
        """
        return super().embed(chips, context)

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
        prompt-relative 0–1 match probability rather than a constant 1.0. The returned
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
        self._contract.chips(chips)
        if not isinstance(queries, list) or any(
            not isinstance(q, str) for q in queries
        ):
            self._contract.fail("queries must be a list of strings.")
        queries = [q for q in queries if q and q.strip()]
        if not queries:
            raise BackendContractError(
                "classify requires at least one non-empty --query."
            )
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
            log_scale = self._contract.tensor(model.logit_scale, (), "logit scale")
            scale = self._contract.tensor(
                log_scale.exp(), (), "exponentiated logit scale"
            )
            text_features = self._contract.tensor(
                model.encode_text(tokens),
                (len(prompts), None),
                "text features",
                nonzero_rows=True,
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            for selection in self.batches(chips.shape[0], context):
                batch = chips[selection]
                pixels = (
                    torch.from_numpy(np.ascontiguousarray(batch))
                    .float()
                    .to(self.device)
                )
                image_features = self._contract.tensor(
                    model.encode_image(pixels),
                    (batch.shape[0], text_features.shape[1]),
                    "image features",
                    nonzero_rows=True,
                )
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )
                logits = scale * (image_features @ text_features.T)
                probs = logits.softmax(dim=-1)[:, :n_q]
                scores.append(probs.cpu().numpy())
        return self._contract.probabilities(
            np.concatenate(scores, axis=0).astype(np.float32), chips.shape[0], n_q
        )
