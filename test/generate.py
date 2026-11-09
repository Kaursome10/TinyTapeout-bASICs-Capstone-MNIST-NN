"""Rebuild mnist_test_images.mem and mnist_test_labels.mem from the raw MNIST test files.

Reads t10k-images-idx3-ubyte.gz and t10k-labels-idx1-ubyte.gz (next to this script),
average-pools every 28x28 image in non-overlapping 4x4 blocks to 7x7 (rounded to the
nearest integer, as in the hardware's input format) and writes, next to this script:

  mnist_test_images.mem   one image per line, 49 two-digit hex pixels separated by spaces
  mnist_test_labels.mem   one label per line

Needs only numpy.  Run from anywhere:  python test/generate.py
"""

import gzip
import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_idx(filename, magic, header_ints):
    """Return the raw payload of a gzipped IDX file as a flat uint8 array plus its dimensions."""
    with gzip.open(os.path.join(HERE, filename), "rb") as f:
        data = f.read()
    values = struct.unpack(f">{header_ints + 1}I", data[:4 * (header_ints + 1)])
    if values[0] != magic:
        raise SystemExit(f"{filename}: unexpected magic number {values[0]}")
    payload = np.frombuffer(data, dtype=np.uint8, offset=4 * (header_ints + 1))
    return payload, values[1:]


images, (count, rows, cols) = read_idx("t10k-images-idx3-ubyte.gz", 2051, 3)
labels, (label_count,) = read_idx("t10k-labels-idx1-ubyte.gz", 2049, 1)
if (rows, cols) != (28, 28) or count != label_count:
    raise SystemExit("unexpected MNIST file layout")

images = images.reshape(count, rows, cols).astype(np.int32)
print("Applying 4x4 block average pooling (28x28 -> 7x7)...")
# Reshape 28x28 into 7x4x7x4 and take the mean across the 4x4 blocks (axes 2 and 4).
pixels = np.rint(images.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)))
pixels = pixels.reshape(-1, 49).astype(np.int32)

print(f"Writing {count} pooled images and labels...")
with open(os.path.join(HERE, "mnist_test_images.mem"), "w") as img_f, \
        open(os.path.join(HERE, "mnist_test_labels.mem"), "w") as lbl_f:
    for i in range(count):
        # Each pixel is a 2-digit hex byte (00-FF).
        img_f.write(" ".join(f"{p:02X}" for p in pixels[i]) + "\n")
        lbl_f.write(f"{labels[i]}\n")

print("Wrote mnist_test_images.mem and mnist_test_labels.mem")
