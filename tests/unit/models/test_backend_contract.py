"""Concrete shared-interface conformance and malformed inference regressions."""

from __future__ import annotations

import numpy as np
import pytest

from whirld.core.fetch import Manifest
from whirld.core.options import InferenceOptions
from whirld.errors import (
    BackendContractError,
    InvalidInputError,
    UnsupportedOperationError,
)
from whirld.models.base import BackendContract, InferenceContext, ModelBackend
from whirld.models.clay import ClayBackend
from whirld.models.clay_torch import ClayTorchBackend
from whirld.models.prithvi import PrithviBackend
from whirld.models.remoteclip import RemoteCLIPBackend


def _reference() -> ClayBackend:
    """Construct a tiny deterministic reference without touching the cache."""
    return ClayBackend(
        Manifest(
            name="reference",
            version="test",
            source_type="reference",
            sha256="0" * 64,
            weights_file="weights.json",
            embed_dim=8,
            seed=1,
        ),
        "cpu",
    )


def _backend(provider: str) -> ModelBackend:
    """Construct an unloaded adapter, deliberately leaving all heavy models absent."""
    if provider == "reference":
        return _reference()
    if provider == "clay":
        return ClayTorchBackend("clay", "test", "cpu", embed_dim=8, patch_size=8)
    if provider == "remoteclip":
        return RemoteCLIPBackend("remoteclip", "test", "cpu", arch="RN50")
    return PrithviBackend("prithvi", "test", "cpu", classes=3)


@pytest.mark.parametrize(
    "provider,operation,supported",
    [
        ("reference", "classify", ("embed",)),
        ("reference", "segment", ("embed",)),
        ("clay", "classify", ("embed",)),
        ("clay", "segment", ("embed",)),
        ("remoteclip", "embed", ("classify",)),
        ("remoteclip", "segment", ("classify",)),
        ("prithvi", "embed", ("segment",)),
        ("prithvi", "classify", ("segment",)),
    ],
)
def test_unsupported_direct_operations_are_cheap_domain_errors(
    provider: str,
    operation: str,
    supported: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported operations ignore malformed inputs and never import a runtime."""
    backend = _backend(provider)
    import builtins

    original = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        """Fail if capability reporting acquires a heavyweight runtime."""
        assert name.split(".")[0] not in {"torch", "open_clip", "terratorch"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(UnsupportedOperationError) as raised:
        if operation == "classify":
            backend.classify(None, [])  # type: ignore[arg-type]
        else:
            getattr(backend, operation)(None)
    error = raised.value
    assert isinstance(error, InvalidInputError) and error.exit_code == 7
    assert (error.model, error.operation, error.supported_operations) == (
        backend.name,
        operation,
        supported,
    )
    assert backend.supported_operations == supported


@pytest.mark.parametrize("provider", ["reference", "clay", "remoteclip", "prithvi"])
@pytest.mark.parametrize(
    "malformation", ["rank", "dtype", "nonfinite", "zero-height", "zero-bands"]
)
def test_bad_input_rejected_before_loading(provider: str, malformation: str) -> None:
    """All adapters reject invalid normalized arrays without acquiring weights."""
    backend = _backend(provider)
    channels = 3 if provider == "remoteclip" else 6
    chips = np.ones((1, channels, 8, 8), dtype=np.float32)
    if malformation == "rank":
        chips = chips[0]
    elif malformation == "dtype":
        chips = chips.astype(np.float64)
    elif malformation == "nonfinite":
        chips[0, 0, 0, 0] = np.nan
    elif malformation == "zero-height":
        chips = chips[:, :, :0]
    else:
        chips = chips[:, :0]
    with pytest.raises(BackendContractError):
        if provider == "remoteclip":
            backend.classify(chips, ["field"])
        elif provider == "prithvi":
            backend.segment(chips)
        else:
            backend.embed(chips)


@pytest.mark.parametrize(
    "provider,channels,shape,dtype",
    [
        ("reference", 6, (0, 8), "float32"),
        ("clay", 6, (0, 8), "float32"),
        ("remoteclip", 3, (0, 1), "float32"),
        ("prithvi", 6, (0, 8, 8), "uint8"),
    ],
)
def test_empty_valid_batches_do_not_load(
    provider: str,
    channels: int,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    """Valid empty requests retain documented output contracts without a model."""
    backend = _backend(provider)
    chips = np.empty((0, channels, 8, 8), dtype=np.float32)
    result = (
        backend.classify(chips, ["field"])
        if provider == "remoteclip"
        else backend.segment(chips) if provider == "prithvi" else backend.embed(chips)
    )
    assert result.shape == shape and result.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "metadata",
    [
        {"embed_dim": 0},
        {"embed_dim": True},
        {"patch_size": -1},
        {"classes": 256},
        {"classes": 1.5},
        {"channels": 0},
    ],
)
def test_invalid_metadata_is_a_domain_error(metadata: dict[str, object]) -> None:
    """Typed metadata cannot truncate dimensions or admit an unrepresentable mask."""
    with pytest.raises(BackendContractError):
        BackendContract.create("bad", **metadata)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 7), np.float32),
        np.ones((1, 8), np.float32),
        np.ones((2, 8), np.float64),
        np.full((2, 8), np.inf, np.float32),
    ],
)
def test_embedding_output_contract(bad: np.ndarray) -> None:
    """Output dimensions, row count, dtype and finiteness are independently enforced."""
    with pytest.raises(BackendContractError):
        BackendContract.create("encoder", embed_dim=8).embeddings(bad, 2)


def test_probability_and_mask_contracts() -> None:
    """Single prompt scores need no unit sum, while multiclass masks stay bounded."""
    contract = BackendContract.create("head", classes=3)
    single = np.array([[0.2]], np.float32)
    assert contract.probabilities(single, 1, 1) is single
    for scores in [np.array([[0.2, 0.3]], np.float32), np.array([[1.1]], np.float32)]:
        with pytest.raises(BackendContractError):
            contract.probabilities(scores, 1, scores.shape[1])
    with pytest.raises(BackendContractError):
        contract.masks(np.array([[[3]]], np.uint8), (1, 1, 1))


class _CustomBackend(ModelBackend):
    """Custom providers implement the shared interface without registration."""

    def embed(
        self, chips: np.ndarray, context: InferenceContext | None = None
    ) -> np.ndarray:
        """Return a trivial custom embedding."""
        return np.empty((len(chips), 1), np.float32)


def test_custom_backends_need_no_catalog_registration() -> None:
    """Catalog declarations do not block existing injected/custom subclasses."""
    backend = _CustomBackend("custom", "1", "cpu")
    assert backend.embed(np.empty((0, 1, 1, 1))).shape == (0, 1)
    with pytest.raises(UnsupportedOperationError, match="compatible model"):
        backend.segment(np.empty((0, 1, 1, 1)))


@pytest.mark.parametrize(
    "fault", ["rows", "classes", "spatial", "integer", "nonfinite", "not-tensor"]
)
def test_prithvi_raw_logits_and_recovery(fault: str) -> None:
    """Reject malformed logits before argmax and preserve request batching."""
    torch = pytest.importorskip("torch")

    class Head:
        """Inject one bad result, then recover to a valid three-class task."""

        broken = True
        seen: list[int]

        def __init__(self) -> None:
            self.seen = []

        def __call__(self, pixels: object) -> object:
            n, _, _, h, w = pixels.shape
            self.seen.append(n)
            shape = [n, 3, h, w]
            if self.broken:
                if fault == "not-tensor":
                    return None
                if fault in {"rows", "classes", "spatial"}:
                    shape[{"rows": 0, "classes": 1, "spatial": 2}[fault]] += 1
                result = torch.zeros(
                    shape, dtype=torch.int32 if fault == "integer" else torch.float32
                )
                if fault == "nonfinite":
                    result[0, 0, 0, 0] = float("nan")
                return result
            result = torch.zeros(shape)
            result[:, 2] = 1
            return result

    head = Head()
    backend = PrithviBackend("head", "test", "cpu", classes=3, model=head, batch_size=2)
    chips = np.ones((5, 6, 8, 8), np.float32)
    context = InferenceContext("test", 10, options=InferenceOptions(batch_size=1))
    with pytest.raises(BackendContractError):
        backend.segment(chips, context=context)
    head.broken = False
    head.seen.clear()
    masks = backend.segment(chips)
    assert head.seen == [2, 2, 1]
    assert masks.shape == (5, 8, 8) and masks.dtype == np.uint8 and np.all(masks == 2)


@pytest.mark.parametrize(
    "fault",
    ["text-rows", "image-rows", "image-dim", "zero", "nonfinite", "integer", "scale"],
)
def test_remoteclip_raw_features(fault: str) -> None:
    """Reject malformed CLIP features before normalization or softmax."""
    torch = pytest.importorskip("torch")

    class Clip:
        logit_scale = torch.tensor(float("inf") if fault == "scale" else 0.0)

        def encode_text(self, tokens: object) -> object:
            return torch.ones((3 if fault == "text-rows" else len(tokens), 4))

        def encode_image(self, pixels: object) -> object:
            result = torch.ones(
                (
                    len(pixels) + (fault == "image-rows"),
                    5 if fault == "image-dim" else 4,
                ),
                dtype=torch.int32 if fault == "integer" else torch.float32,
            )
            if fault == "zero":
                result[:] = 0
            if fault == "nonfinite":
                result[0, 0] = float("nan")
            return result

    backend = RemoteCLIPBackend(
        "clip",
        "test",
        "cpu",
        arch="RN50",
        model=Clip(),
        tokenizer=lambda prompts: torch.zeros((len(prompts), 1)),
    )
    with pytest.raises(BackendContractError):
        backend.classify(np.ones((2, 3, 8, 8), np.float32), ["field"])


@pytest.mark.parametrize(
    "fault", ["rows", "dim", "no-cls", "integer", "nonfinite", "not-tuple"]
)
def test_clay_raw_encoder_contract(fault: str) -> None:
    """Check token dimensions, CLS presence and numeric validity before extraction."""
    torch = pytest.importorskip("torch")

    class Encoder:
        def __call__(self, cube: object) -> object:
            if fault == "not-tuple":
                return None
            result = torch.ones(
                (
                    len(cube["pixels"]) + (fault == "rows"),
                    0 if fault == "no-cls" else 2,
                    9 if fault == "dim" else 8,
                ),
                dtype=torch.int32 if fault == "integer" else torch.float32,
            )
            if fault == "nonfinite":
                result[0, 0, 0] = float("inf")
            return (result,)

    backend = ClayTorchBackend(
        "clay", "test", "cpu", embed_dim=8, patch_size=8, encoder=Encoder()
    )
    with pytest.raises(BackendContractError):
        backend.embed(
            np.ones((2, 6, 8, 8), np.float32),
            InferenceContext("test", 10, wavelengths=[0.5] * 6),
        )


@pytest.mark.parametrize(
    "field,value",
    [("embed_dim", None), ("embed_dim", 0), ("patch_size", None), ("patch_size", True)],
)
def test_clay_direct_constructor_metadata(field: str, value: object) -> None:
    """Direct callers receive domain errors for missing or malformed dimensions."""
    metadata = {"embed_dim": 8, "patch_size": 8, field: value}
    with pytest.raises(BackendContractError):
        ClayTorchBackend("clay", "test", "cpu", **metadata)


@pytest.mark.parametrize("classes", [None, 0, 256, True, 1.5])
def test_prithvi_direct_constructor_metadata(classes: object) -> None:
    """Reject missing/non-integral or unrepresentable direct class declarations."""
    with pytest.raises(BackendContractError):
        PrithviBackend("head", "test", "cpu", classes=classes)


@pytest.mark.parametrize("provider,channels", [("remoteclip", 4), ("prithvi", 3)])
def test_fixed_input_channel_count(provider: str, channels: int) -> None:
    """RGB and six-band adapters enforce their respective channel contracts."""
    backend = _backend(provider)
    with pytest.raises(BackendContractError):
        if provider == "remoteclip":
            backend.classify(np.zeros((1, channels, 8, 8), np.float32), ["field"])
        else:
            backend.segment(np.zeros((1, channels, 8, 8), np.float32))
