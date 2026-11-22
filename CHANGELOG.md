# Changelog

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
