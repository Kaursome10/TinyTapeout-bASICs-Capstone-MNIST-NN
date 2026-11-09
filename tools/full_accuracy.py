"""Run MNIST test images through the actual Verilog and report the accuracy.

Compiles src/project.v and src/nn_core.v with sim/tb_full.v in Icarus Verilog, streams
the weights and the 7x7 test images (test/*.mem) through the Tiny Tapeout pins, and
compares the chip's digit for every image with the labels and with the bit-exact
Python model in tools/evaluate_mnist.py.

Run from anywhere:  python tools/full_accuracy.py
Quick check:        python tools/full_accuracy.py --num-images 200

All 10,000 images take several minutes. Needs iverilog (and vvp) on the PATH.
Exits with a non-zero status if the chip disagrees with the model, drops `ready`
outside the two layer boundaries, or fails to raise `done`.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import evaluate_mnist as model

ROOT_DIR = model.ROOT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-images", type=int, default=10000,
                        help="how many of the 10,000 test images to run (default: all)")
    parser.add_argument("--weights-dir", default=model.DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--test-dir", default=model.DEFAULT_TEST_DIR)
    args = parser.parse_args()

    for tool in ("iverilog", "vvp"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found. Install Icarus Verilog (for example: sudo apt-get install iverilog).")

    pixel_file = os.path.join(args.test_dir, "mnist_test_images.mem")
    labels = model.load_labels(os.path.join(args.test_dir, "mnist_test_labels.mem"))[:args.num_images]
    pixels = model.load_images(pixel_file)[:args.num_images]
    if not 0 < len(labels) <= 10000:
        sys.exit("--num-images must be between 1 and 10000")

    # Python reference model (same arithmetic as src/nn_core.v).
    w1 = model.load_weights(os.path.join(args.weights_dir, "weights_W1.mem"), (model.HID1, 49))
    w2 = model.load_weights(os.path.join(args.weights_dir, "weights_W2.mem"), (model.HID2, model.HID1))
    w3 = model.load_weights(os.path.join(args.weights_dir, "weights_W3.mem"), (10, model.HID2))
    hidden1 = model.requantize(pixels @ w1.T, model.SHIFTS[0])
    hidden2 = model.requantize(hidden1 @ w2.T, model.SHIFTS[1])
    expected = (hidden2 @ w3.T).argmax(axis=1)

    with tempfile.TemporaryDirectory() as tmp:
        vvp_file = os.path.join(tmp, "sim.vvp")
        out_file = os.path.join(tmp, "digits.txt")
        subprocess.run(
            ["iverilog", "-g2012", "-o", vvp_file,
             os.path.join(ROOT_DIR, "src", "project.v"),
             os.path.join(ROOT_DIR, "src", "nn_core.v"),
             os.path.join(ROOT_DIR, "sim", "tb_full.v")],
            check=True)
        print(f"Simulating {len(labels)} images (this can take several minutes)...", flush=True)
        run = subprocess.run(
            ["vvp", vvp_file,
             f"+W1={os.path.join(args.weights_dir, 'weights_W1.mem')}",
             f"+W2={os.path.join(args.weights_dir, 'weights_W2.mem')}",
             f"+W3={os.path.join(args.weights_dir, 'weights_W3.mem')}",
             f"+PIX={pixel_file}", f"+OUT={out_file}", f"+N={len(labels)}"],
            capture_output=True, text=True)
        sys.stdout.write(run.stdout)
        sys.stderr.write(run.stderr)
        if run.returncode != 0 or not os.path.exists(out_file):
            sys.exit("simulation failed")
        with open(out_file) as f:
            chip = np.array([int(line) for line in f if line.strip()], dtype=np.int64)

    summary = re.search(r"stall_errors=(\d+) done_timeouts=(\d+)", run.stdout)
    if summary is None or len(chip) != len(labels):
        sys.exit(f"simulation did not finish: got {len(chip)} of {len(labels)} results")
    stall_errors, done_timeouts = int(summary.group(1)), int(summary.group(2))

    correct = np.count_nonzero(chip == labels)
    mismatches = np.count_nonzero(chip != expected)
    print(f"images={len(labels)}")
    print(f"accuracy={correct / len(labels):.4%}")
    print(f"correct={correct}/{len(labels)}")
    print(f"chip_vs_model_mismatches={mismatches}")
    print(f"ready_errors={stall_errors} done_timeouts={done_timeouts}")

    if mismatches or stall_errors or done_timeouts:
        sys.exit(1)


if __name__ == "__main__":
    main()
