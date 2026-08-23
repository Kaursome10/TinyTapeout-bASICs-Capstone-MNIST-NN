<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

This project implements a 3-layer quantized neural network that classifies
7x7 grayscale handwritten digit images (a downsampled MNIST). Pixels are
loaded serially through ld_addr/ld_data before inference. The network runs
49 -> 64 -> 32 -> 10, using matvec_top_banked for each layer's matrix-vector
multiply, ReLU and requantization between layers, and an argmax at the end
to pick the predicted digit.

## How to test

Load all 49 pixel values by pulsing ld_en once per pixel while driving
ld_addr from 0 to 48 and ld_data with the corresponding pixel value. Once
all pixels are loaded, pulse start for one cycle. Wait for valid to go
high, then read the predicted digit (0-9) from out[3:0].

## External hardware

None. All inputs and outputs are digital signals used for loading pixel
data and reading back the classification result — no external peripherals
are required to operate this project.