# Clean-room install + smoke test for Whirld (PRD section 17.3).
#
# Builds a fresh Linux x86 image with ONLY the published package + its base
# dependencies (no dev tools, no test harness, no pre-built virtualenv), then runs
# scripts/clean_room_test.py to prove the shipped surface works end to end —
# including the STAC item reader against rasterio's bundled GDAL.
#
# Build + run (offline-hermetic checks):
#   docker build -f docker/clean-room.Dockerfile -t whirld-cleanroom .
#   docker run --rm whirld-cleanroom
#
# Add a real remote STAC item (live /vsicurl/ range reads):
#   docker run --rm \
#     -e WHIRLD_TEST_STAC_URL="https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/<id>" \
#     -e WHIRLD_TEST_STAC_BBOX="<min_lon,min_lat,max_lon,max_lat>" \
#     whirld-cleanroom

FROM python:3.13-slim

# System GDAL is intentionally NOT installed — Whirld relies on rasterio's bundled
# GDAL (PRD section 15.5). This image proves that holds on a stock Linux base.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WHIRLD_HOME=/tmp/whirld-home

WORKDIR /app

# rasterio's bundled GDAL needs a couple of shared libs not present in -slim
# (libexpat for XML; this is the minimal set the wheel does not vendor). System
# GDAL itself is still NOT installed.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Install the package (base deps only — the STAC item path needs no extra).
# Copy just what the build needs so a source edit doesn't bust the dep layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install .

# The clean-room smoke test (depends only on the installed package).
COPY scripts/clean_room_test.py ./clean_room_test.py

CMD ["python", "clean_room_test.py"]
