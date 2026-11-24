"""Verified small Sentinel-2 sample shipped with the installed quickstart."""

from __future__ import annotations

import shutil
from importlib.resources import as_file, files
from pathlib import Path
from typing import Annotated, Literal

import rasterio
from pydantic import BaseModel, ConfigDict, Field
from rasterio.errors import RasterioError

from ..core.fetch import sha256_file
from ..errors import InvalidInputError
from ..io.raster import validate_metric_grid


class DemoSampleSpec(BaseModel):
    """Frozen file identity, spectral units, grid and licensed source lineage."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1
    id: str
    file: Literal["starter-rgb.tif"] = "starter-rgb.tif"
    sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    size_bytes: int = Field(strict=True, gt=0, le=16 * 1024 * 1024)
    width: Literal[448] = 448
    height: Literal[448] = 448
    crs: str
    transform: tuple[float, float, float, float, float, float]
    band_names: tuple[Literal["B04"], Literal["B03"], Literal["B02"]] = (
        "B04",
        "B03",
        "B02",
    )
    sensor: Literal["sentinel-2-l2a"] = "sentinel-2-l2a"
    dtype: Literal["float32"] = "float32"
    units: Literal["canonical_dn"] = "canonical_dn"
    nodata: float = Field(default=-9999.0, strict=True, ge=-9999.0, le=-9999.0)
    source_item_url: str
    source_item_id: str
    source_bbox: tuple[float, float, float, float]
    source_datetime: str
    source_crop_window: tuple[int, int, int, int]
    attribution: str
    license_url: str
    preparation: str


class BundledDemoSample:
    """Copy only hash-verified packaged TIFF bytes into a fresh owned run."""

    @staticmethod
    def load() -> DemoSampleSpec:
        """Read the immutable packaged metadata without source imagery requests.

        Returns:
            Source identity and exact native RGB raster contract.

        Raises:
            InvalidInputError: Packaged metadata is missing or malformed.
        """
        try:
            resource = files("whirld.demo").joinpath("data/sample.json")
            return DemoSampleSpec.model_validate_json(resource.read_bytes())
        except (OSError, ValueError) as exc:
            raise InvalidInputError(
                "Packaged demo sample metadata is invalid."
            ) from exc

    def prepare(self, output_dir: Path) -> tuple[DemoSampleSpec, Path]:
        """Verify/copy the installed sample and require its declared grid and units.

        Args:
            output_dir: Existing directory newly reserved by the demo runner.

        Returns:
            Verified sample metadata and a standalone local TIFF copy.

        Raises:
            InvalidInputError: Packaged bytes or GTiff header differ from the pin.
            OSError: A sample copy cannot be created without overwriting a file.
        """
        spec = self.load()
        target = output_dir / spec.file
        resource = files("whirld.demo").joinpath("data", spec.file)
        with as_file(resource) as source:
            if (
                source.stat().st_size != spec.size_bytes
                or sha256_file(source) != spec.sha256
            ):
                raise InvalidInputError(
                    "Packaged demo sample checksum differs from pin."
                )
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        if sha256_file(target) != spec.sha256:
            raise InvalidInputError("Copied demo sample checksum differs from pin.")
        try:
            # The sample is one pinned TIFF. Associated PAM/mask/world files are
            # outside that byte identity and must not influence its interpretation.
            with (
                rasterio.Env(
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    GDAL_PAM_ENABLED=False,
                    GDAL_GEOREF_SOURCES="INTERNAL",
                ),
                rasterio.open(target, driver="GTiff") as dataset,
            ):
                if (
                    dataset.count != 3
                    or dataset.shape != (spec.height, spec.width)
                    or dataset.dtypes != (spec.dtype,) * 3
                    or dataset.nodata != spec.nodata
                    or dataset.crs is None
                    or dataset.crs != rasterio.crs.CRS.from_user_input(spec.crs)
                    or tuple(dataset.transform)[:6] != spec.transform
                    or tuple(dataset.descriptions) != spec.band_names
                    or validate_metric_grid(spec.crs, dataset.transform) != (10.0, 10.0)
                ):
                    raise InvalidInputError(
                        "Packaged demo raster header differs from pin."
                    )
        except RasterioError as exc:
            raise InvalidInputError(
                "Packaged demo sample is not a valid GeoTIFF."
            ) from exc
        return spec, target
