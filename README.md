# Description

A simple rust extension for blending together image frames from a bounded queue.

## Setup

### CPython with pip & virtualenv

```bash
virtualenv .venv && \
source .venv/bin/activate && \
pip install -r requirements.txt && \
cd rust_ext && \
maturin develop --release && \
cd -
```

### CPython with conda

```bash
conda env create -f environment.yml && \
conda activate blend-frames && \
cd rust_ext && \
maturin develop --release && \
cd -
```

### PyPy with conda

```bash
conda env create -f environment-pypy.yml && \
conda activate blend-frames-pypy && \
cd rust_ext && \
maturin develop --release && \
cd -
```

## Usage

### Use the rust extension from a .py file

```python
import os
import numpy as np
from PIL import Image

import rust_ext

MAX_ITERS = 1_000
filepaths = [
    os.path.join(os.getcwd(), "frames", f)
    for f in os.listdir("frames")
    if f.endswith(".jpg")
]
images = [Image.open(fpath) for fpath in filepaths]
frames = np.array([np.asarray(im) for im in images])

print(rust_ext.average(frames))
print(rust_ext.blend_frames(frames, MAX_ITERS))
```

### Benchmark the performance of the rust extension

```bash
python benchmarks.py
```
