"""Fresh worker RSS remains bounded while windowed GeoTIFF output grows."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel


class WriterMeasurement(BaseModel):
    """A completed output's pixel/byte count and process-lifetime peak bytes."""

    pixels: int
    output_bytes: int
    rss: int


def test_segmentation_writer_memory_plateaus_as_output_grows(tmp_path: Path) -> None:
    """Writers generate bounded tile batches without allocating a scene mask."""
    worker = """
import json,resource,sys
from pathlib import Path
import numpy as np
from whirld.core.chips import ChipMeta
from whirld.io.segmentation_stream import SegmentationMetadata,SegmentationStreamWriter
size=int(sys.argv[1]);tile=256
meta=SegmentationMetadata(crs='EPSG:32630',
 transform=(10.0,0.0,500000.0,0.0,-10.0,4500000.0),
 width=size,height=size,chip_size_px=tile,classes=2)
cols=(size+tile-1)//tile
count=cols*cols
rng=np.random.default_rng(244)
with SegmentationStreamWriter(Path(sys.argv[2]),chip_count=count,metadata=meta) as w:
 for start in range(0,count,4):
  chips=[]
  for i in range(start,min(start+4,count)):
   row,col=divmod(i,cols)
   x=500000+10*col*tile;y=4500000-10*row*tile
   chips.append(ChipMeta(index=i,row=row,col=col,
    bbox=[float(x),float(y-10*tile),float(x+10*tile),float(y)]))
  masks=rng.integers(0,2,(len(chips),tile,tile),dtype=np.uint8)
  validity=np.ones(masks.shape,dtype=bool)
  w.write(masks,chips,validity)
 w.finish()
 assert w.valid_pixel_count==size*size
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 sys.stdout.write(json.dumps({'pixels':w.valid_pixel_count,
  'output_bytes':w.output_path.stat().st_size,
  'rss':int(peak if sys.platform=='darwin' else peak*1024)}))
"""
    results = []
    for size in (2048, 4096, 8192):
        completed = subprocess.run(
            [sys.executable, "-c", worker, str(size), str(tmp_path / f"{size}.tif")],
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        measured = WriterMeasurement.model_validate_json(completed.stdout)
        assert measured.pixels == size * size
        results.append(measured)
    assert results[2].output_bytes > 3 * results[1].output_bytes
    assert results[1].rss - results[0].rss < 32 * 1024**2, results
    assert results[2].rss - results[1].rss < 32 * 1024**2, results
