# Prithvi flood segmentation

`prithvi-flood` is the official Prithvi EO 2.0 300M TL model fine-tuned on
Sen1Floods11. It segments **water/flood**, not flood change relative to a dry-date
baseline. Class 0 is no water and class 1 is water/flood. Whirld writes source
invalid pixels as 255 so valid class 0 remains distinguishable from nodata.

Install the `prithvi` extra using the project's installation instructions, then:

```bash
whirld pull prithvi-flood
whirld segment projected-l1c.tif --model prithvi-flood \
  --sensor sentinel-2-l1c --device cpu --output flood-mask.tif
# Equivalent model alias: --model prithvi-eo-2 --head flood
```

The checkpoint is 1,276,843,350 bytes (about 1.28 GB), downloaded separately from
the package. The model is Apache-2.0; its model card/config/inference source are
published by IBM/NASA:

- [Official model card](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11)
- Pinned revision: `91ce9d38086a80b078a192b374df758b8855b732`.
- Checkpoint: `Prithvi-EO-V2-300M-TL-Sen1Floods11.pt`.
- SHA-256: `76eed77d8bd543ae441308b80e8408502243a871cb8a338ddc8220ed96dfc270`.
- Deployment config: `config.yaml`, pinned to the same revision. Whirld verifies
  installed artifacts and disables training/pretrained-backbone bootstrapping.

## Explicit input contract

Use Sentinel-2 **L1C** raw DN imagery, not an assumed L2A or HLS substitute. The
selected bands, in model order, are B02, B03, B04, B8A, B11, B12 (blue, green, red,
narrow NIR, SWIR1, SWIR2). Each is interpreted as reflectance after multiplying DN
by 0.0001 and standardized using the published Sen1Floods11 datamodule statistics:

| Band | Mean | Standard deviation |
|---|---:|---:|
| B02 | 0.1412956 | 0.07406382 |
| B03 | 0.13795798 | 0.07370365 |
| B04 | 0.12353792 | 0.08692279 |
| B8A | 0.30902815 | 0.11798815 |
| B11 | 0.2044958 | 0.09772074 |
| B12 | 0.11912015 | 0.07659938 |

The target grid is 10 meters with 512 × 512 native chips. TIFF/STAC scale/offset
metadata is interpreted through Whirld's existing physical-reflectance conversion;
for already-scaled direct COG pixels, explicitly declare `CogOptions(scale=1.0)`.
L1C processing level is a distinct sensor identity; select it explicitly when
source metadata does not identify the processing level unambiguously.

### Temporal/location policy

The backbone has optional time/location conditioning, but the pinned fine-tuning
configuration explicitly sets `use_metadata: false`. Whirld follows that
**metadata-free** training path and calls the segmentation model without optional
TL coordinates. It does not infer acquisition times from filenames or inject
location coordinates simply because they are available in the API context.

The upstream demonstration script makes a different optional choice: it supplies
scene coordinates from `rasterio.lnglat()` (longitude/latitude), whereas the
backbone documents location input as latitude/longitude. It also uses reflective
edge padding. Whirld uses the same normalized-zero padding as its other inference
workflows. Therefore literal upstream demo masks are not an equality oracle for
Whirld's supported metadata/padding policy. The opt-in comparison below explicitly
uses the same metadata-free policy, projected pixels and normalized-zero padding.
The upstream script also scales its saved visualization prediction from 0/1 to
0/255 and declares nodata 0. Whirld preserves raw class indices 0/1 and reserves
255 for source-invalid pixels; comparisons use raw argmax masks, not the saved
upstream TIFF bytes.

## Small official example

The official India example is a 13-band geographic-grid TIFF. Whirld's grid
contract requires a north-up projected metric grid. Reproject it explicitly before
inference; `--crs` only assigns a missing CRS and does not perform reprojection.
The following prepares a six-band metric fixture from pinned source bytes:

```python
import hashlib
from pathlib import Path
from urllib.request import urlretrieve

import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

url = (
    'https://huggingface.co/ibm-nasa-geospatial/'
    'Prithvi-EO-2.0-300M-TL-Sen1Floods11/resolve/'
    '91ce9d38086a80b078a192b374df758b8855b732/'
    'examples/India_900498_S2Hand.tif'
)
source = Path('India_900498_S2Hand.tif')
urlretrieve(url, source)
assert hashlib.sha256(source.read_bytes()).hexdigest() == (
    'ee898621b387a731503a01960397599f209b45c268d4e088689928c704dbe968'
)
with rasterio.open(source) as src:
    affine, width, height = calculate_default_transform(
        src.crs, 'EPSG:32646', src.width, src.height, *src.bounds, resolution=10
    )
    profile = src.profile.copy()
    profile.update(crs='EPSG:32646', transform=affine, width=width, height=height,
                   count=6, compress='lzw')
    with rasterio.open('projected-l1c.tif', 'w', **profile) as dst:
        for output_band, source_band in enumerate([2, 3, 4, 9, 12, 13], start=1):
            reproject(
                source=rasterio.band(src, source_band),
                destination=rasterio.band(dst, output_band),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=affine, dst_crs='EPSG:32646',
                src_nodata=0, dst_nodata=0, resampling=Resampling.nearest,
            )
        dst.descriptions = ('B02', 'B03', 'B04', 'B8A', 'B11', 'B12')
```

This prepares a fixture 461 pixels wide by 513 pixels high, exercising a partial final chip row. Use the
CLI command above or `whirld.segment(..., model='prithvi-flood',
sensor='sentinel-2-l1c', device='cpu', output='flood-mask.tif')`.

## Verification and limitations

Unit tests independently check the pinned registry, exact L1C normalization,
band reordering, masks and rejection of HLS/L2A overrides. The opt-in actual-weight
test checks independent stored-DN normalization and raw model inference against
Whirld's model path, with all network access denied after installation:

```bash
WHIRLD_TEST_FLOOD_HOME=/path/to/whirld-home \
WHIRLD_TEST_FLOOD_SAMPLE=/path/to/projected-l1c.tif \
  .venv/bin/python -m pytest tests/integration/models/test_prithvi_flood_real.py
```

The real comparison passed on Python 3.13.7 with the pinned TerraTorch dependencies;
metadata-free raw reference and backend masks were exactly equal. A public API
run on the projected India fixture produced a 513 × 461 mask with 233029 valid
pixels and 109762 predicted water/flood pixels in 7.06 seconds on the development
CPU (two threads, cached weights). On the first chip, supplying the upstream
optional longitude/latitude location changed 138 predictions relative to the
metadata-free policy; supplying the backbone-documented latitude/longitude changed
177. These measurements establish that the optional policy choice is material.
This verifies execution/contract fidelity rather than flood-mapping accuracy. Source-cloud
masks must be supplied by the input; the segmentation head does not make every
cloud an automatically invalid input pixel. Reprojection and temporal change
analysis remain separate operations.
