# Changelog

## Unreleased

- Add Prithvi EO 2.0 flood segmentation with an explicit Sentinel-2 L1C
  contract, pinned official checkpoint/config and documented metadata/padding
  policy. Predicted water/flood is class 1; valid background is 0, nodata is 255.
- Add `segment_to_disk()` and `segment --stream` for native-tile bounded-memory
  mask GeoTIFF output, retaining source validity and exact target grid/extent.
- Add pinned RemoteCLIP RN50 and ViT-L/14 registry entries alongside existing
  ViT-B/32, with actual preprocessing/checkpoint and eager/stream validation.
- Require explicit L1C selection or unambiguous processing-level metadata for
  flood detection; shared Sentinel-2 band names cannot distinguish L1C from L2A.

No release/tag is created by these changes. PRs are manually merged by the user,
and release publication requires separate explicit approval.

## 0.2.0 — 2026-11-22

- Add `classify_to_disk()` and CLI `classify --stream` for bounded STAC/direct
  COG classification with one backend and configurable batch/cache budgets.
- Preserve eager GeoJSON fields, inclusive thresholds, global score ordering and
  stable ties. `top_k=None` / `--all` emits every match using disk-backed sorting.
- Stage completed features in SQLite and atomically publish the single GeoJSON
  only after successful completion; failures preserve prior output.
- Add classification streaming documentation, numerical/HTTP/CLI/failure and
  memory checks, plus real RemoteCLIP eager/stream validation.

Validation: 704 tests passed, 9 opt-in checks skipped, 91.53% package/example
coverage; fresh wheel/source installs pass. Real RemoteCLIP batches 1 and 3 retain
all 16 pinned-scene chips with matching geometry/ranking and maximum score error
below 5.4e-6. Writer-only RSS remains about 70–71 MiB while output grows from
4,000 to 64,000 features (1.55–24.96 MB); raster memory is tested separately.

Existing eager classification defaults and REST behavior remain compatible.
Streaming requires a disk output and STAC/direct remote input; temporary disk
space grows with retained features. North-up/same-CRS requirements continue;
segmentation streaming and reprojection #21 remain deferred.

## 0.1.0 — 2026-11-22

First release of Whirld, a local-first CLI and Python library for geospatial
foundation-model inference. APIs and output contracts may evolve during 0.x;
future releases will document compatibility changes here.

### Included

- Sensor-aware band selection, radiometry, mask/nodata handling, resampling,
  chip geometry and acquisition/location metadata.
- Real Clay v1.5 embeddings, RemoteCLIP text classification and Prithvi burn-scar
  segmentation, with separately installed model extras and verified checkpoints.
- An offline `clay-v1` deterministic reference encoder for setup/tests; use
  `clay-v1.5` for trained Clay model embeddings.
- Local GeoTIFF, STAC item and direct HTTP(S) COG inputs in the library/CLI,
  including bbox windows, explicit source band/unit metadata and scoped auth.
- Bounded-memory embedding output to NPY plus JSON metadata, using window reads
  and incremental writes.
- CLI model management, a REST server for uploaded rasters, registry governance,
  checkpoint integrity verification and local usage records.
- Three runnable examples using pinned public imagery and real model weights.

### Installation and verification

Download the wheel or source distribution from the GitHub release. Use Python
3.13.7 or newer; verification uses 3.13.7 on macOS ARM64 and Ubuntu 24.04 x86-64 CI.
Other Python versions/platforms have not been verified. Base installation uses
Rasterio's bundled GDAL. Real models require extras and separate checkpoint
retrieval (Clay v1.5: 5.16 GB; RemoteCLIP: 605 MB; Prithvi burn-scar: 1.30 GB).
Whirld is Apache-2.0; model weights and vendored code retain their upstream
licenses/notices. See README and `whirld info MODEL` before downloading.

Release baseline: 605 tests passed, 9 opt-in checks skipped, 90.98% coverage across
the package and examples. Additional public-data and real-model CPU runs verified
streaming/eager equivalence and all three examples; these are execution checks,
not claims of prediction accuracy or GPU performance.

### Known limitations

- Inputs require north-up projected grids in meters. Streaming STAC assets must
  share a CRS; reprojection is deferred in issue #21.
- Only embeddings support bounded-memory output. Classification and segmentation
  retain the selected region in memory; use a bbox for smaller eager reads.
- Remote rasters require HTTP range support. Tiled COGs are recommended; source
  strips can enlarge decoder buffers. Direct rasters require explicit sensor and
  unambiguous source band identity.
- REST endpoints accept uploads; remote-input request bodies and REST streaming
  remain deferred. STAC search/mosaics, quantization, flood segmentation and remote
  registry synchronization also remain deferred.
- NPY/JSON output publication uses per-file replacement and rollback; the pair is
  not crash-atomic. Concurrent runs must use distinct output destinations.
- PyPI publication is separate from this GitHub release.
