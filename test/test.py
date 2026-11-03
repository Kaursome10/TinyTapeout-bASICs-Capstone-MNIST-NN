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
PIXEL_SIGNED = True       # must match nn_core's PIXEL_SIGNED parameter


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
    for pixels in test_images().values():
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
