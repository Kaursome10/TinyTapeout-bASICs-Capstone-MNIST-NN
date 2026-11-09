"""Evaluate the checked-in quantized weights on all 10,000 MNIST test images.

Bit-exact model of the hardware (src/nn_core.v): 49 -> 40 -> 20 -> 10, signed
int4 weights, ReLU, rounded shifts of 7 and 4, and a 0..63 clamp on the hidden
activations. Inputs are the 7x7 average-pooled images in test/*.mem, which
test/generate.py produces from the MNIST test set.

Run from anywhere:  python tools/evaluate_mnist.py
"""

import argparse
import os

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEIGHTS_DIR = os.path.join(ROOT_DIR, "src", "weights")
DEFAULT_TEST_DIR = os.path.join(ROOT_DIR, "test")

HID1, HID2 = 40, 20
ABITS = 6
SHIFTS = (7, 4)
AMAX = (1 << ABITS) - 1


def load_weights(filename, shape):
    """One hex nibble per line, two's complement int4, row-major."""
    with open(filename) as f:
        values = np.array([int(line.strip(), 16) for line in f if line.strip()],
                          dtype=np.int64)
    if values.size != shape[0] * shape[1]:
        raise SystemExit(f"{filename}: expected {shape[0] * shape[1]} weights, got {values.size}")
    values = np.where(values >= 8, values - 16, values)
    return values.reshape(shape)


def requantize(values, shift):
    """ReLU, add rounding constant, shift right, clamp to 0..AMAX."""
    values = np.maximum(values, 0)
    return np.minimum((values + (1 << (shift - 1))) >> shift, AMAX)


def load_images(filename):
    with open(filename) as f:
        return np.array([[int(token, 16) for token in line.split()]
                         for line in f if line.strip()], dtype=np.int64)


def load_labels(filename):
    with open(filename) as f:
        return np.array([int(line) for line in f if line.strip()], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--test-dir", default=DEFAULT_TEST_DIR,
                        help="folder with mnist_test_images.mem and mnist_test_labels.mem")
    args = parser.parse_args()

    weights = [
        load_weights(os.path.join(args.weights_dir, "weights_W1.mem"), (HID1, 49)),
        load_weights(os.path.join(args.weights_dir, "weights_W2.mem"), (HID2, HID1)),
        load_weights(os.path.join(args.weights_dir, "weights_W3.mem"), (10, HID2)),
    ]
    pixels = load_images(os.path.join(args.test_dir, "mnist_test_images.mem"))
    labels = load_labels(os.path.join(args.test_dir, "mnist_test_labels.mem"))

    hidden1 = requantize(pixels @ weights[0].T, SHIFTS[0])
    hidden2 = requantize(hidden1 @ weights[1].T, SHIFTS[1])
    scores = hidden2 @ weights[2].T
    predictions = scores.argmax(axis=1)  # first maximum wins, like the hardware

    correct = np.count_nonzero(predictions == labels)
    all_weights = np.concatenate([weight.ravel() for weight in weights])
    print(f"accuracy={correct / len(labels):.4%}")
    print(f"correct={correct}/{len(labels)}")
    print(f"nonzero_weights={np.count_nonzero(all_weights)}/{all_weights.size}")
    print(f"zero_weights={np.count_nonzero(all_weights == 0)}/{all_weights.size}")


if __name__ == "__main__":
    main()