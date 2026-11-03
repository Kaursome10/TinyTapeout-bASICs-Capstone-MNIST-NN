# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

IMG_SIZE = 49
MAX_WAIT_CYCLES = 30000

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# TODO: once you have a real labeled MNIST sample, replace PIXELS with its
# 49 pixel values (0-255) and set EXPECTED_DIGIT to the known correct label.
PIXELS = [0] * IMG_SIZE
EXPECTED_DIGIT = None

def load_weights(filename):
    path = os.path.join(WEIGHTS_DIR, filename)
    with open(path) as f:
        return [int(line.strip(), 16) if int(line.strip(), 16) < 128
                else int(line.strip(), 16) - 256
                for line in f if line.strip()]


def build_uio(addr: int = 0, data_valid: int = 0, start: int = 0, weight: int = 0) -> int:
    if data_valid and weight:
        return ((weight & 0xF) << 4) | ((data_valid & 1) << 1) | (start & 1)
    return ((addr & 0x3F) << 2) | ((data_valid & 1) << 1) | (start & 1)


async def stream_layer_weights(dut, weights, boundary_cycles=0, pixels=None):
    """Stream one row-major matrix, then wait for its pending final row commit."""
    for index, w in enumerate(weights):
        dut.ui_in.value = (pixels[index % IMG_SIZE] if pixels else 0) & 0xFF
        dut.uio_in.value = build_uio(data_valid=1, weight=w)
        await RisingEdge(dut.clk)

    dut.uio_in.value = build_uio(addr=0, data_valid=0, start=0)
    if boundary_cycles:
        await ClockCycles(dut.clk, boundary_cycles)


@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)

    dut._log.info("Pulsing start")
    dut.uio_in.value = build_uio(addr=0, data_valid=0, start=1)
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = build_uio(addr=0, data_valid=0, start=0)

    dut._log.info("Streaming layer 1 weights")
    await stream_layer_weights(dut, load_weights("weights_W1.mem"), 2, PIXELS)

    dut._log.info("Streaming layer 2 weights")
    await stream_layer_weights(dut, load_weights("weights_W2.mem"), 2)

    dut._log.info("Streaming layer 3 weights")
    await stream_layer_weights(dut, load_weights("weights_W3.mem"))

    dut._log.info("Waiting for valid")
    for _ in range(MAX_WAIT_CYCLES):
        await ClockCycles(dut.clk, 1)
        if (int(dut.uo_out.value) >> 4) & 1:
            break
    else:
        assert False, f"valid never asserted within {MAX_WAIT_CYCLES} cycles"

    digit = int(dut.uo_out.value) & 0xF
    dut._log.info(f"Predicted digit: {digit}")

    assert 0 <= digit <= 9, f"out={digit} is outside the valid digit range"

    if EXPECTED_DIGIT is not None:
        assert digit == EXPECTED_DIGIT, f"expected {EXPECTED_DIGIT}, got {digit}"