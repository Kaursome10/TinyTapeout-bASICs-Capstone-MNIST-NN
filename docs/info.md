<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This project classifies 7x7 grayscale handwritten digits (downsampled MNIST)
with a small quantized neural network: 49 inputs -> 50 -> 25 -> 10 outputs,
with int4 weights and 7-bit activations.

To fit in 2x2 tiles, the chip does not store the weights or the image.
Instead, the host streams them in while the chip computes. A single
multiply-accumulate unit works through the three layers in order, one
weight per clock cycle:

| Layer | Size    | Each element the host sends            | Elements |
|-------|---------|----------------------------------------|----------|
| 1     | 49 -> 50 | pixel `x[k]` + weight `W1[i][k]`       | 2450     |
| 2     | 50 -> 25 | weight `W2[i][k]`                      | 1250     |
| 3     | 25 -> 10 | weight `W3[i][k]`                      | 250      |

Elements are sent row by row (for each output `i`, all inputs `k`). At the
end of each row of layers 1 and 2, the sum goes through ReLU, is divided by a
power of two with rounding (2^6 after layer 1, 2^5 after layer 2), clamped to
0..127 and stored on chip. Layer 3's ten scores are compared as they finish,
and the index of the highest one is the predicted digit.

Weights are signed 4-bit values (-8..7). Pixels are signed 8-bit values
(-128..127), so keep pixel values in 0..127.

## How to test

1. Reset the chip.
2. Pulse `start` (uio[0]) high for one clock cycle. `ready` (uo[5]) goes high
   and `layer` (uo[7:6]) shows 1.
3. For each element, in the order above: set `weight` (uio[7:4]), and during
   layer 1 also `pixel` (ui[7:0]), set `in_valid` (uio[1]) high, and give one
   clock edge. The host may pause at any time by holding `in_valid` low.
4. After the 3950th element, `ready` goes low, and within two cycles `done`
   (uo[4]) goes high. Read the predicted digit from uo[3:0].
5. `done` and the digit stay put until the next `start`. Pulsing `start` in the
   middle of an inference abandons it and begins a new one.

Change inputs away from the rising clock edge (for example on the falling
edge, or by stepping the clock from the host), since inputs are sampled on the
rising edge.

The weight files are in `src/weights/` (one hex nibble per line, row-major),
and `test/test.py` contains a Python reference model plus a complete example
of the streaming protocol.

## External hardware

None. A host (such as the Tiny Tapeout demo board's microcontroller) drives
the inputs and reads back the result.
