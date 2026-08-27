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

COMPUTE_STATE = 1  # matvec_top_streamed's localparam COMPUTE = 1


def load_weights(filename):
    path = os.path.join(WEIGHTS_DIR, filename)
    with open(path) as f:
        return [int(line.strip(), 16) if int(line.strip(), 16) < 128
                else int(line.strip(), 16) - 256
                for line in f if line.strip()]


def build_uio(addr: int, data_valid: int, start: int) -> int:
    return ((addr & 0x3F) << 2) | ((data_valid & 1) << 1) | (start & 1)


async def stream_layer_weights(dut, layer_inst, weights):
    """Wait for a specific matvec_top_streamed instance to enter COMPUTE,
    then feed its weights one per cycle. Drain-phase gaps (where no layer
    is computing) are skipped by design -- no pulses are sent, and hence
    none are dropped, since we only assert weight_valid once this exact
    layer is actively listening."""
    while int(layer_inst.state.value) != COMPUTE_STATE:
        await RisingEdge(dut.clk)

    for w in weights:
        dut.ui_in.value = w & 0xFF
        dut.uio_in.value = build_uio(addr=0, data_valid=1, start=0)
        await RisingEdge(dut.clk)

    dut.uio_in.value = build_uio(addr=0, data_valid=0, start=0)


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

    dut._log.info("Loading pixels")
    for addr in range(IMG_SIZE):
        dut.ui_in.value = PIXELS[addr] & 0xFF
        dut.uio_in.value = build_uio(addr=addr, data_valid=1, start=0)
        await ClockCycles(dut.clk, 1)

    dut.uio_in.value = build_uio(addr=0, data_valid=0, start=0)
    await ClockCycles(dut.clk, 1)

    dut._log.info("Pulsing start")
    dut.uio_in.value = build_uio(addr=0, data_valid=0, start=1)
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = build_uio(addr=0, data_valid=0, start=0)

    nn = dut.user_project.nn

    dut._log.info("Streaming layer 1 weights")
    await stream_layer_weights(dut, nn.h1, load_weights("weights_W1.mem"))

    dut._log.info("Streaming layer 2 weights")
    await stream_layer_weights(dut, nn.h2, load_weights("weights_W2.mem"))

    dut._log.info("Streaming layer 3 weights")
    await stream_layer_weights(dut, nn.h3, load_weights("weights_W3.mem"))

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