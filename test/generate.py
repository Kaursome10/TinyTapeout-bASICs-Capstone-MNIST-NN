"""Generate mnist_test_images.mem and mnist_test_labels.mem using 
exact torchvision MNIST loading and 4x4 block average pooling."""

import os
import numpy as np
from torchvision.datasets import MNIST

print("Loading MNIST test set via torchvision to match golden model...")
dataset = MNIST("/tmp/mnist-data", train=False, download=True)
images = dataset.data.numpy().astype(np.int32)
labels = dataset.targets.numpy().astype(np.int32)

print("Applying exact 4x4 block average pooling (matching evaluate_mnist.py)...")
# Reshape 28x28 into 7x4x7x4 and take the mean across the 4x4 blocks (axes 2 and 4)
pixels = np.rint(images.reshape(-1, 7, 4, 7, 4).mean(axis=(2, 4)))
pixels = pixels.reshape(-1, 49).astype(np.int32)

print(f"Writing {len(images)} properly pooled samples to .mem files...")
with open("mnist_test_images.mem", "w") as img_f, open("mnist_test_labels.mem", "w") as lbl_f:
    for i in range(len(images)):
        # Format each pixel as a 2-digit hex byte (00-FF)
        img_f.write(" ".join(f"{p:02X}" for p in pixels[i]) + "\n")
        lbl_f.write(f"{labels[i]}\n")

print("Successfully generated mnist_test_images.mem and mnist_test_labels.mem!")