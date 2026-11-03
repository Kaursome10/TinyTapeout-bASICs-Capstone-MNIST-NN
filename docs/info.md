<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This project implements a 3-layer quantized neural network that classifies
7x7 grayscale handwritten digit images (a downsampled MNIST). The network runs
49 -> 50 -> 25 -> 10 with signed int4 weights, unsigned 8-bit pixels, 7-bit
hidden activations, ReLU, and rounded requantization shifts of 6 and 5. A
single serialized MAC processes all layers, while the host streams each pixel
alongside its layer-1 weight. The checked-in quantized weights measure 95.21%
accuracy on the standard 10,000-image MNIST test split.
The compact streamed implementation maps to approximately 25,482 um^2 in the
sky130 standard-cell area estimate. A Tiny Tapeout 2x2 area is approximately
334 x 225 = 75,150 um^2, so this estimate uses about 34% of the available area
before placement and routing overhead. The configured clock period remains
52 ns because timing analysis is not available in the lightweight local flow.

## How to test

Pulse start for one cycle, then stream the 3,950 signed 4-bit weights in
row-major order with data_valid high. During the first 2,450 weight cycles,
drive ui_in with the corresponding 8-bit pixel; drive uio[7:4] with the
weight nibble. Wait for valid to go high, then read the predicted digit
(0-9) from out[3:0].

## External hardware

None. All inputs and outputs are digital signals used for loading pixel
data and reading back the classification result — no external peripherals
are required to operate this project.