# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0
#
# Tests for the compact MNIST NN. Only the top-level pins are used, so the
# same test runs on the RTL and on the gate-level netlist.
#
# Every result is checked against a bit-exact Python model of the network.

import os
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "weights")

# (rows, cols, weight file) for each layer, weights stored row-major
LAYERS = [
    (50, 49, "weights_W1.mem"),
    (25, 50, "weights_W2.mem"),
    (10, 25, "weights_W3.mem"),
]
REQUANT_SHIFTS = [6, 5]   # after layer 1 and layer 2, must match nn_core's R1_SHIFT / R2_SHIFT
PIXEL_SIGNED = False      # must match nn_core's PIXEL_SIGNED parameter

# First 20 digits of the MNIST test set, shrunk to 7x7 by averaging each 4x4
# block (pixel values 0..255). Format: (label, pixels row by row).
MNIST_SAMPLES = [
    (7, [0, 0, 0, 0, 0, 0, 0, 0, 17, 25, 0, 0, 0, 0, 0, 41, 100, 125, 156, 75, 0, 0, 0, 0, 0, 146, 8, 0, 0, 0, 0, 60, 88, 0, 0, 0, 0, 10, 154, 2, 0, 0, 0, 0, 66, 81, 0, 0, 0]),
    (2, [0, 0, 15, 52, 6, 0, 0, 0, 8, 177, 149, 83, 0, 0, 0, 0, 3, 166, 37, 0, 0, 0, 0, 92, 112, 0, 0, 0, 0, 0, 195, 15, 2, 22, 19, 0, 0, 144, 176, 150, 123, 58, 0, 0, 0, 0, 0, 0, 0]),
    (1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 100, 0, 0, 0, 0, 0, 58, 60, 0, 0, 0, 0, 0, 121, 5, 0, 0, 0, 0, 0, 131, 0, 0, 0, 0, 0, 39, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 210, 37, 0, 0, 0, 2, 186, 193, 185, 29, 0, 0, 16, 168, 1, 74, 143, 0, 0, 54, 135, 73, 216, 61, 0, 0, 9, 184, 224, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4, [0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 20, 26, 0, 0, 5, 118, 0, 61, 63, 0, 0, 56, 63, 0, 146, 4, 0, 0, 13, 110, 100, 182, 0, 0, 0, 0, 0, 0, 155, 0, 0, 0, 0, 0, 0, 15, 0, 0]),
    (1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 116, 0, 0, 0, 0, 0, 85, 113, 0, 0, 0, 0, 0, 164, 25, 0, 0, 0, 0, 7, 175, 0, 0, 0, 0, 0, 41, 115, 0, 0, 0, 0, 0, 2, 16, 0, 0, 0]),
    (4, [0, 0, 0, 0, 0, 0, 0, 0, 1, 100, 2, 0, 69, 0, 0, 67, 80, 0, 82, 59, 0, 0, 44, 149, 115, 174, 0, 0, 0, 0, 0, 87, 52, 0, 0, 0, 0, 0, 159, 58, 0, 0, 0, 0, 0, 24, 2, 0, 0]),
    (9, [0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 54, 0, 0, 0, 0, 0, 152, 201, 53, 0, 0, 0, 0, 147, 149, 125, 0, 0, 0, 0, 49, 35, 140, 9, 0, 0, 0, 0, 0, 27, 106, 0, 0, 0, 0, 0, 0, 49, 0]),
    (5, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 98, 188, 186, 45, 0, 37, 88, 51, 24, 1, 0, 0, 165, 59, 18, 7, 0, 0, 0, 57, 153, 234, 207, 0, 0, 0, 0, 4, 175, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (9, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 33, 0, 0, 0, 8, 140, 134, 163, 153, 0, 0, 51, 174, 156, 221, 88, 0, 0, 0, 3, 180, 76, 0, 0, 0, 0, 92, 142, 0, 0, 0, 0, 0, 119, 10, 0, 0, 0]),
    (0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 143, 158, 51, 0, 0, 2, 138, 0, 8, 149, 0, 0, 49, 70, 0, 0, 138, 0, 0, 85, 50, 0, 43, 120, 0, 0, 32, 173, 172, 154, 5, 0, 0, 0, 0, 0, 0, 0, 0]),
    (6, [0, 0, 13, 27, 0, 0, 0, 0, 0, 120, 27, 0, 0, 0, 0, 1, 131, 18, 121, 92, 0, 0, 41, 105, 160, 27, 102, 0, 0, 52, 108, 115, 61, 91, 0, 0, 2, 109, 132, 79, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (9, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 29, 5, 0, 0, 0, 94, 118, 150, 54, 0, 0, 0, 184, 67, 172, 0, 0, 0, 0, 55, 107, 104, 0, 0, 0, 0, 0, 132, 21, 0, 0, 0, 0, 0, 92, 0, 0, 0]),
    (0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 150, 157, 0, 0, 0, 0, 185, 63, 127, 67, 0, 0, 21, 163, 0, 29, 132, 0, 0, 38, 135, 0, 42, 108, 0, 0, 0, 119, 60, 148, 13, 0, 0, 0, 1, 39, 12, 0, 0]),
    (1, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 175, 12, 0, 0, 0, 0, 0, 183, 24, 0, 0, 0, 0, 0, 191, 16, 0, 0, 0, 0, 0, 183, 16, 0, 0, 0, 0, 8, 187, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (5, [0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 124, 68, 3, 0, 0, 0, 134, 68, 5, 0, 0, 0, 5, 78, 41, 104, 53, 0, 0, 0, 0, 0, 0, 124, 0, 0, 10, 132, 99, 101, 74, 0, 0, 0, 0, 0, 0, 0, 0]),
    (9, [0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 110, 29, 0, 0, 0, 14, 142, 7, 119, 40, 0, 0, 57, 81, 9, 177, 7, 0, 0, 2, 106, 101, 156, 0, 0, 0, 0, 0, 67, 99, 0, 0, 0, 0, 0, 58, 22, 0, 0]),
    (7, [0, 0, 0, 0, 0, 0, 0, 0, 52, 52, 19, 0, 0, 0, 0, 76, 101, 138, 178, 48, 0, 0, 0, 0, 0, 113, 65, 0, 0, 0, 0, 31, 178, 2, 0, 0, 0, 4, 187, 31, 0, 0, 0, 0, 54, 88, 0, 0, 0]),
    (3, [0, 0, 0, 0, 0, 0, 0, 0, 106, 179, 60, 0, 0, 0, 0, 128, 27, 174, 56, 0, 0, 0, 34, 125, 197, 188, 84, 0, 0, 0, 135, 7, 4, 174, 26, 0, 0, 43, 140, 159, 160, 10, 0, 0, 0, 0, 0, 0, 0]),
    (4, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 66, 0, 0, 0, 0, 97, 0, 103, 0, 0, 0, 1, 159, 101, 146, 0, 0, 0, 0, 39, 15, 135, 0, 0, 0, 0, 0, 37, 83, 0, 0, 0, 0, 0, 7, 12, 0, 0]),
]


# ---------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------
def load_weights(filename, count):
    """Each line is one hex nibble holding a signed int4 (-8..7)."""
    values = []
    with open(os.path.join(WEIGHTS_DIR, filename)) as f:
        for line in f:
            s = line.strip()
            if s:
                v = int(s, 16)
                values.append(v - 16 if v >= 8 else v)
    assert len(values) == count, f"{filename}: expected {count} weights, got {len(values)}"
    return values


def load_all_weights():
    return [load_weights(fname, rows * cols) for rows, cols, fname in LAYERS]


def requant(acc, shift):
    """ReLU, round, divide by 2^shift, clamp to 0..127."""
    acc = max(acc, 0)
    return min((acc + (1 << (shift - 1))) >> shift, 127)


def reference(pixels, weights):
    """Returns (digit, scores) exactly as the hardware computes them."""
    x = [(p - 256 if (PIXEL_SIGNED and p >= 128) else p) for p in pixels]
    for layer, (rows, cols, _) in enumerate(LAYERS):
        w = weights[layer]
        sums = [sum(w[i * cols + k] * x[k] for k in range(cols)) for i in range(rows)]
        if layer < 2:
            x = [requant(s, REQUANT_SHIFTS[layer]) for s in sums]
    scores = sums
    return scores.index(max(scores)), scores  # first max wins, like the hardware


# ---------------------------------------------------------------------------
# Pin helpers (inputs change on the falling edge, sampled on the rising edge)
# ---------------------------------------------------------------------------
def uio(start=0, valid=0, weight=0):
    return ((weight & 0xF) << 4) | ((valid & 1) << 1) | (start & 1)


def outputs(dut):
    out = int(dut.uo_out.value)
    return {
        "digit": out & 0xF,
        "done": (out >> 4) & 1,
        "ready": (out >> 5) & 1,
        "layer": (out >> 6) & 3,
    }


async def reset(dut):
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await FallingEdge(dut.clk)


async def pulse_start(dut):
    dut.ui_in.value = 0
    dut.uio_in.value = uio(start=1)
    await FallingEdge(dut.clk)
    dut.uio_in.value = uio()


async def stream_elements(dut, pixels, weights, rng=None, stall_prob=0.0, limit=None):
    """Send elements in order. Stops early after `limit` elements if given."""
    sent = 0
    for layer, (rows, cols, _) in enumerate(LAYERS):
        w = weights[layer]
        for i in range(rows):
            for k in range(cols):
                if limit is not None and sent >= limit:
                    dut.uio_in.value = uio()
                    return
                while rng is not None and rng.random() < stall_prob:
                    dut.uio_in.value = uio()
                    await FallingEdge(dut.clk)
                o = outputs(dut)
                assert o["ready"] == 1, f"ready dropped at layer {layer + 1} row {i} col {k}"
                assert o["layer"] == layer + 1, f"expected layer {layer + 1}, chip says {o['layer']}"
                assert o["done"] == 0, "done went high before all elements were sent"
                dut.ui_in.value = pixels[k] if layer == 0 else 0
                dut.uio_in.value = uio(valid=1, weight=w[i * cols + k])
                await FallingEdge(dut.clk)
                sent += 1
    dut.ui_in.value = 0
    dut.uio_in.value = uio()


async def wait_done(dut, max_cycles=10):
    for _ in range(max_cycles):
        if outputs(dut)["done"]:
            return
        await FallingEdge(dut.clk)
    assert False, f"done never went high within {max_cycles} cycles"


async def run_image(dut, pixels, weights, rng=None, stall_prob=0.0):
    await pulse_start(dut)
    await stream_elements(dut, pixels, weights, rng, stall_prob)
    await wait_done(dut)

    expected, scores = reference(pixels, weights)
    got = outputs(dut)["digit"]
    dut._log.info(f"scores={scores} expected={expected} got={got}")
    assert got == expected, f"expected digit {expected}, got {got}"

    # Result must stay put (done held high) and the chip must stop accepting.
    for _ in range(20):
        await FallingEdge(dut.clk)
        o = outputs(dut)
        assert o["done"] == 1, "done dropped before the next start"
        assert o["digit"] == expected, "digit changed while done was high"
        assert o["ready"] == 0 and o["layer"] == 0, "chip still accepting after finishing"
    return got


# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------
def test_images():
    rng = random.Random(1234)
    images = {
        "all zeros": [0] * 49,
        "vertical stroke (like a 1)": [100 if (k % 7) == 3 else 0 for k in range(49)],
        "ring (like a 0)": [
            100 if (r in (1, 5) and 2 <= c <= 4) or (c in (1, 5) and 2 <= r <= 4) else 0
            for r in range(7) for c in range(7)
        ],
    }
    for n in range(4):
        images[f"random 0..127 #{n}"] = [rng.randrange(128) for _ in range(49)]
    images["random 0..255"] = [rng.randrange(256) for _ in range(49)]
    return images


@cocotb.test()
async def test_reference_model_ranges(dut):
    """Sanity check the model itself: sums fit in the 18-bit accumulator."""
    weights = load_all_weights()
    worst = 0
    all_images = list(test_images().values()) + [p for _, p in MNIST_SAMPLES]
    all_images.append([255] * 49)  # brightest possible image
    for pixels in all_images:
        x = [(p - 256 if (PIXEL_SIGNED and p >= 128) else p) for p in pixels]
        for layer, (rows, cols, _) in enumerate(LAYERS):
            w = weights[layer]
            sums = [sum(w[i * cols + k] * x[k] for k in range(cols)) for i in range(rows)]
            worst = max(worst, max(abs(s) for s in sums))
            if layer < 2:
                x = [requant(s, REQUANT_SHIFTS[layer]) for s in sums]
    dut._log.info(f"largest |sum| seen: {worst}")
    assert worst < (1 << 17)


@cocotb.test()
async def test_images_back_to_back(dut):
    """Several images in a row, no reset in between, no pauses in the stream."""
    weights = load_all_weights()
    await reset(dut)

    o = outputs(dut)
    assert o["ready"] == 0 and o["done"] == 0, "chip should be idle after reset"

    for name, pixels in test_images().items():
        dut._log.info(f"image: {name}")
        await run_image(dut, pixels, weights)


@cocotb.test()
async def test_real_mnist_digits(dut):
    """Real handwritten digits: chip must match the model exactly, and the
    model gets 19 of these 20 right (91.2% over the full 10,000-digit test set)."""
    weights = load_all_weights()
    await reset(dut)
    correct = 0
    for n, (label, pixels) in enumerate(MNIST_SAMPLES):
        got = await run_image(dut, pixels, weights)
        correct += got == label
        dut._log.info(f"MNIST #{n}: label {label}, chip says {got}")
    dut._log.info(f"chip classified {correct}/{len(MNIST_SAMPLES)} real digits correctly")
    assert correct >= 19


@cocotb.test()
async def test_random_pauses(dut):
    """The host may pause (in_valid low) at any point without changing the result."""
    weights = load_all_weights()
    rng = random.Random(99)
    await reset(dut)
    for name, pixels in list(test_images().items())[1:4]:
        dut._log.info(f"image with pauses: {name}")
        await run_image(dut, pixels, weights, rng=rng, stall_prob=0.3)


@cocotb.test()
async def test_restart_mid_run(dut):
    """A start pulse in the middle of an inference cleanly begins a new one."""
    weights = load_all_weights()
    images = list(test_images().values())
    await reset(dut)

    await pulse_start(dut)
    await stream_elements(dut, images[1], weights, limit=2700)  # stop partway into layer 2

    await run_image(dut, images[2], weights)
