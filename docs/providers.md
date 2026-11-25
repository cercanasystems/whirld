# Model provider contracts

Whirld's seven registered models use four trusted providers: the deterministic
Clay reference backend, the trained Clay v1.5 encoder, RemoteCLIP and Prithvi.
They share `ModelBackend`, including methods that a particular provider does not
support. Keeping those methods is intentional: capability declarations and
consistent early errors define the boundary.

## Ownership and execution

`ProviderCatalog` is a closed catalog of immutable Pydantic provider definitions.
Each definition owns its operations, source/category/output compatibility,
runtime dependencies, optional extra, metadata requirements and lazy constructor.
Provider subclasses encapsulate the policies that differ between families.
Registry documents cannot supply Python import paths or external architecture
selectors.

`ModelEntry` delegates provider metadata validation and operation checks to that
definition. `load_backend()` validates metadata, holds the owning `ModelStore`'s
installation lease and delegates construction to the definition. `doctor` uses
the same dependency and extra declarations without loading a model. Registry and
CLI help imports remain independent of numerical and optional model runtimes.

`ModelBackend` owns the common inference interface and per-call batching policy.
Each concrete backend owns an immutable `BackendContract`, which checks its
documented inputs and actual outputs. Framework construction and tensor
adaptation remain inside the relevant backend.

| Provider | Operation | Model-specific requirements |
|---|---|---|
| `clay-reference` | `embed` | Configurable positive embedding width and deterministic nonnegative seed; NumPy only. |
| `clay` | `embed` | Clay v1.5 large: 1024 dimensions, patch size 8 and finite positive per-band wavelengths. |
| `remoteclip` | `classify` | RGB input and one of the local RN50, ViT-B/32 or ViT-L/14 architectures; text prompts supplied per request. |
| `prithvi` | `segment` | Six optical bands, trusted declared checkpoint/config and 1–255 output classes; uint8 label 255 is reserved for nodata. |

## Unsupported operations

For example, asking `clay-v1` to classify an image raises
`UnsupportedOperationError`, identifies the requested operation and lists the
supported operations. It derives from `InvalidInputError`: CLI exit code **7**,
HTTP **422**, and the same typed exception for Python callers.

The public eager APIs check the selected entry before automatic device selection,
installation reads, imagery reads or preprocessing. Streaming owners defer device
selection until after the same capability check. Session APIs check the actual
resident entry when available and recheck the selected entry when loading or
reusing a model. They do not authorize a task on an old cached model using a
newly edited registry declaration.

REST performs metadata inspection in a worker before reading the uploaded file
into memory or decoding it. Framework multipart parsing can still spool the HTTP
upload before that service check. Requests remain subject to normal argument and
source-routing validation. Explicit session construction or server startup may
select a device or preload models independently of any subsequent request.

Direct calls to unsupported concrete backend methods use the same exception and
do not validate arrays, import optional runtimes or initialize networks first.
The generic session `get()` can still load a provider without choosing a task;
callers can pass `operation=` to require the same early guard.

## Metadata and output validation

Provider metadata is validated when registry entries are constructed, before
acquisition. The loader validates it again before taking an installation lease,
covering programmatically altered entries. Known architecture dimensions,
spectral requirements, channel counts and output declarations must agree.

Backends validate normalized float32 NCHW chips before inference. Learned results
are checked before operations such as CLS extraction, normalization or argmax can
hide an incorrect shape or class count. Returned embeddings and scores must be
finite float32 arrays with the expected dimensions; masks must be uint8 class
indices within the declared vocabulary. Invalid contracts raise
`BackendContractError`, also an input-domain error with CLI 7 / HTTP 422. It
retains `ValueError` compatibility for existing direct callers.

RemoteCLIP scores remain relative to the requested prompts, not calibrated
land-cover confidence. A single query includes an internal contrasting background
prompt; its omitted background column means the returned row need not sum to one.
Validation does not alter the model's scoring or preprocessing arithmetic.

## Adding support

For another checkpoint using a supported provider and architecture, add a registry
entry satisfying that provider's contract and the existing trust/source policy.
For a new family, implement `ModelBackend`, add a trusted provider definition to
the closed catalog and update the schema's backend enum. Their parity is tested.
Package optional dependencies through the existing extras mechanism. Loader,
registry and doctor dispatch should use the provider definition rather than grow
independent backend maps.

Retain shared conformance checks for supported/unsupported operations, lazy
construction, metadata rejection, output shape/dtype, batching and recovery.
Public boundary tests must prove unsupported requests avoid expensive work across
eager, session, streaming, CLI and REST paths. Verify supported workflows with
real checkpoints as well as injected unit-test networks.
