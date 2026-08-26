# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

IMG_SIZE = 49          # 7x7 downsampled image
MAX_WAIT_CYCLES = 15000  # generous upper bound for the full 3-layer pipeline

# TODO: once you have a real labeled MNIST sample and have run it through
# your trained model offline, replace PIXELS with that sample's 49 pixel
# values (0-255) and set EXPECTED_DIGIT to the known correct label. Until
# then, this test only checks that the pipeline completes and produces a
# value in the valid 0-9 range -- it does not check classification accuracy.
PIXELS = [0] * IMG_SIZE
EXPECTED_DIGIT = None  # set to an int (0-9) once you have ground truth


def build_uio(addr: int, ld_en: int, start: int) -> int:
    return ((addr & 0x3F) << 2) | ((ld_en & 1) << 1) | (start & 1)


@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    # Reset
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 1)

    # Load all 49 pixels: drive ld_data on ui_in, address + ld_en on uio_in
    dut._log.info("Loading pixels")
    for addr in range(IMG_SIZE):
        dut.ui_in.value = PIXELS[addr]
        dut.uio_in.value = build_uio(addr=addr, ld_en=1, start=0)
        await ClockCycles(dut.clk, 1)

    # De-assert ld_en before pulsing start
    dut.uio_in.value = build_uio(addr=0, ld_en=0, start=0)
    await ClockCycles(dut.clk, 1)

    # Pulse start for exactly one cycle
    dut._log.info("Pulsing start")
    dut.uio_in.value = build_uio(addr=0, ld_en=0, start=1)
    await ClockCycles(dut.clk, 1)
    dut.uio_in.value = build_uio(addr=0, ld_en=0, start=0)

    # Wait for valid (uo_out[4]) to go high, or time out
    dut._log.info("Waiting for valid")
    for cycle in range(MAX_WAIT_CYCLES):
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