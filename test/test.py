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

def load_mnist_dataset():
    images = []
    with open(os.path.join(os.path.dirname(__file__), "mnist_test_images.mem"), "r") as f:
        for line in f:
            if line.strip():
                pixels = [int(val, 16) for val in line.strip().split()]
                images.append(pixels)
                
    labels = []
    with open(os.path.join(os.path.dirname(__file__), "mnist_test_labels.mem"), "r") as f:
        for line in f:
            if line.strip():
                labels.append(int(line.strip()))
                
    return images, labels

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
    dut._log.info("Start Full MNIST Evaluation")

    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    # Load the 10k dataset
    test_images, test_labels = load_mnist_dataset()
    total_samples = len(test_images)
    correct_predictions = 0

    dut._log.info(f"Loaded {total_samples} test samples. Starting simulation...")

    for i in range(total_samples):
        sample_pixels = test_images[i]
        expected_digit = test_labels[i]

        # Hardware Reset Sequence
        dut.ena.value = 1
        dut.ui_in.value = 0
        dut.uio_in.value = 0
        dut.rst_n.value = 0
        await ClockCycles(dut.clk, 10)
        dut.rst_n.value = 1
        await ClockCycles(dut.clk, 1)

        # Pulse Start
        dut.uio_in.value = build_uio(addr=0, data_valid=0, start=1)
        await ClockCycles(dut.clk, 1)
        dut.uio_in.value = build_uio(addr=0, data_valid=0, start=0)

        # Stream Weights & Pixels
        await stream_layer_weights(dut, load_weights("weights_W1.mem"), 2, sample_pixels)
        await stream_layer_weights(dut, load_weights("weights_W2.mem"), 2)
        await stream_layer_weights(dut, load_weights("weights_W3.mem"))

        # Wait for valid flag
        valid_received = False
        for _ in range(MAX_WAIT_CYCLES):
            await ClockCycles(dut.clk, 1)
            if (int(dut.uo_out.value) >> 4) & 1:
                valid_received = True
                break
        
        assert valid_received, f"Sample {i}: valid never asserted"

        digit = int(dut.uo_out.value) & 0xF

        if digit == expected_digit:
            correct_predictions += 1

        # Log progress every 100 samples
        if (i + 1) % 100 == 0:
            current_acc = (correct_predictions / (i + 1)) * 100
            dut._log.info(f"Processed {i + 1}/{total_samples} | Current Accuracy: {current_acc:.2f}%")

    final_accuracy = (correct_predictions / total_samples) * 100
    dut._log.info(f"=== Final MNIST Test Accuracy: {final_accuracy:.2f}% ({correct_predictions}/{total_samples}) ===")