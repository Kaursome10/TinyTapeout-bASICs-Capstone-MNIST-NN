"""Evaluate the checked-in quantized weights on all 10,000 MNIST test images."""

import argparse
import os
import numpy as np
from torchvision.datasets import MNIST

# Determine the project main directory (one level up from this tools directory)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEIGHTS_DIR = os.path.join(ROOT_DIR, "src", "weights")


def load_weights(filename, shape):
    values = np.array([int(line.strip(), 16) for line in open(filename)
                       if line.strip()], dtype=np.int32)
    values = np.where(values >= 8, values - 16, values)
    return values.reshape(shape)


def requantize(values, shift):
    values = np.maximum(values, 0)
    return np.minimum((values + (1 << (shift - 1))) >> shift, 127)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--data-dir", default="/tmp/mnist-data")
    args = parser.parse_args()

    weights = [
        load_weights(os.path.join(args.weights_dir, "weights_W1.mem"), (50, 49)),
        load_weights(os.path.join(args.weights_dir, "weights_W2.mem"), (25, 50)),
        load_weights(os.path.join(args.weights_dir, "weights_W3.mem"), (10, 25)),
    ]
    
    dataset = MNIST(args.data_dir, train=False, download=True)
    images = dataset.data.numpy().astype(np.int32)
    pixels = np.rint(images.reshape(-1, 7, 4, 7, 4).mean((2, 4)))
    pixels = pixels.reshape(-1, 49).astype(np.int32)

    hidden1 = requantize(pixels @ weights[0].T, 6)
    hidden2 = requantize(hidden1 @ weights[1].T, 5)
    scores = hidden2 @ weights[2].T
    predictions = scores.argmax(axis=1)
    accuracy = np.mean(predictions == dataset.targets.numpy())
    all_weights = np.concatenate([weight.ravel() for weight in weights])

    print(f"accuracy={accuracy:.4%}")
    print(f"correct={np.count_nonzero(predictions == dataset.targets.numpy())}/10000")
    print(f"nonzero_weights={np.count_nonzero(all_weights)}/{all_weights.size}")
    print(f"zero_weights={np.count_nonzero(all_weights == 0)}/{all_weights.size}")


if __name__ == "__main__":
    main()