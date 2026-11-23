"""Fresh local streaming workers retain fixed memory as input/output grows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.integration.test_local_streaming import LocalScene

_WORKER = """
import json,resource,sys
import numpy as np
import whirld
from whirld.core.fetch import Manifest,ModelStore
from whirld.models.base import ModelBackend
import whirld.core.classification_stream as classification
import whirld.core.segmentation_stream as segmentation
class Backend(ModelBackend):
 classes=2
 def embed(self,chips,context=None): raise AssertionError()
 def classify(self,chips,queries,context=None):
  return np.full((len(chips),len(queries)),0.5,dtype=np.float32)
 def segment(self,chips,head=None,threshold=0.5,context=None):
  return (chips[:,0]>threshold).astype(np.uint8)
def manifest(store,name):
 return Manifest(name=name,version=store.registry.get(name).version,
  source_type='huggingface',sha256='0'*64,weights_file='fixture.pt')
def backend(entry,manifest,device=None,**kwargs):
 return Backend(entry.name,manifest.version,'cpu')
operation,source,output=sys.argv[1:]
if operation=='embed':
 whirld.pull('clay-v1')
 result=whirld.embed_to_disk(source,model='clay-v1',output=output,
  chip_size=128,batch_size=4,cache_mb=8,device='cpu')
 count=result.chip_count
else:
 ModelStore.load_manifest=manifest
 classification.load_backend=backend
 segmentation.load_backend=backend
 if operation=='classify':
  result=whirld.classify_to_disk(source,model='remoteclip',
   query=['forest','water'],output=output,top_k=None,batch_size=4,
   cache_mb=8,device='cpu')
  count=result.feature_count
 else:
  result=whirld.segment_to_disk(source,model='prithvi-burn-scar',
   sensor='hls',output=output,batch_size=2,cache_mb=8,device='cpu')
  count=result.width*result.height
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
sys.stdout.write(json.dumps({'count':count,
 'rss':int(peak if sys.platform=='darwin' else peak*1024),
 'bytes':result.output_path.stat().st_size}))
"""


@pytest.mark.parametrize("operation", ["embed", "classify", "segment"])
def test_local_streaming_rss_plateaus(tmp_path: Path, operation: str) -> None:
    """Fixed tile, batch and cache bounds survive 64-fold local scene growth.

    The parent creates all source blocks before the measured fresh interpreter.
    Workers report summary counts and disk sizes without reading entire outputs.
    """
    observations: list[dict[str, int]] = []
    for size in (512, 2048, 4096):
        directory = tmp_path / str(size)
        source = LocalScene(
            directory / "source.tif",
            width=size,
            height=size,
            resolution=(30.0, 30.0) if operation == "segment" else (10.0, 10.0),
            external_mask=True,
        ).create()
        suffix = {"embed": ".npy", "classify": ".geojson", "segment": ".tif"}
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                _WORKER,
                operation,
                str(source),
                str(directory / ("result" + suffix[operation])),
            ],
            env={
                **os.environ,
                "WHIRLD_HOME": str(directory / "home"),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        observed = json.loads(process.stdout)
        expected_count = {
            "embed": ((size + 127) // 128) ** 2,
            "classify": ((size + 223) // 224) ** 2,
            "segment": size**2,
        }[operation]
        assert observed["count"] == expected_count
        observations.append(observed)
    assert observations[2]["bytes"] > 3 * observations[1]["bytes"]
    assert observations[1]["rss"] - observations[0]["rss"] < 64 * 1024**2, observations
    assert observations[2]["rss"] - observations[1]["rss"] < 32 * 1024**2, observations
