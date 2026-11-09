![](../../workflows/gds/badge.svg) ![](../../workflows/docs/badge.svg) ![](../../workflows/test/badge.svg) ![](../../workflows/fpga/badge.svg)
![](./docs/bwsi.png)

<p align="left">
  <img src="docs/MITRE_logo.png" alt="MITRE" width="200"><br>
  <em>MITRE: Proud sponsor of Basics of ASICs</em>
</p>

# MNIST Neural Network ASIC

This project is a tiny ASIC that recognises handwritten digits. It runs a quantized 3-layer neural network
(49 → 40 → 20 → 10) on a 7×7 version of an MNIST image and reports the predicted digit, 0 to 9. The network reaches
**96.21 %** accuracy on the full 10,000-image MNIST test set in RTL simulation. This repository contains the training script,
the Verilog source, the cocotb testbench and the Tiny Tapeout configuration used to tape the design out on a **1×2 tile**
(SKY130). An earlier version of this project kept its weights in an on-chip ROM; this version streams 4-bit weights in so that it fits a 1×2 tile.
We acknowledge the MIT Beaver Works Summer Institute (BWSI) for their guidance and for creating this opportunity.

Project contributors: Simran Kaur, Mihir Kotamraju, Nimay Shah, Franklin Chen

## At a glance

| | |
|---|---|
| Network | 49 → 40 → 20 → 10, fully connected, ReLU, argmax |
| Input | 28×28 MNIST image average-pooled to 7×7, unsigned 8-bit pixels |
| Weights | signed 4-bit, 2,960 in total (11,840 bits), streamed in by a host |
| Compute | one shift-and-add MAC, 18-bit accumulator, one weight per clock |
| Accuracy | 96.21 % (9,621 / 10,000) on the MNIST test set |
| Latency | 2,963 clock cycles from `start` to `done` (about 74 µs at 40 MHz) |
| Tile | Tiny Tapeout 1×2, SKY130, top module `tt_um_kaursome10_run_nn` |
| Memory | 360 flip-flops for hidden activations, no SRAM or other macros |

## Architecture

![Architecture](./docs/architecture.png)

The whole network runs on **one** multiply-accumulate unit that processes one weight per clock cycle.

- **No multiplier.** The weights are only 4 bits, so `activation × weight` is a 16-case table of shifts and adds.
- **Cheap scaling.** Layers 1 and 2 add a rounding constant, shift right by 7 and 4 (powers of two, so no multiplier), and clamp the result to 0…63.
- **Tiny memory.** Only the hidden activations are stored: `h1` is 40 × 6 bits and `h2` is 20 × 6 bits, held in rotating shift registers so no read multiplexer is needed. Layer 3 keeps a running argmax, so the ten raw scores are never stored.
- **Weights come from the outside.** A weight ROM for 2,960 values would not fit a 1×2 tile, so the host streams
  the weights in, one per clock cycle, in row-major order. In layer 1 it sends the matching pixel together with each weight.
- **Flow control.** The `ready` output tells the host when an element is accepted. It drops for one cycle after layer 1 and after layer 2, while the last neuron of that layer is stored.

The sizes are set by parameters in `src/nn_core.v`:

| Layer | Shape (neurons × inputs) | Weights | After the layer |
|---|---|---|---|
| 1 | 40 × 49 | 1,960 | ReLU, round, `>> 7`, clamp to 0…63, store in `h1` |
| 2 | 20 × 40 | 800 | ReLU, round, `>> 4`, clamp to 0…63, store in `h2` |
| 3 | 10 × 20 | 200 | running argmax over the raw scores |

## Interface and protocol

| Pin | Name | Direction | Meaning |
|---|---|---|---|
| `ui_in[7:0]` | pixel | in | Pixel for the current weight (layer 1 only) |
| `uio[0]` | start | in | One-cycle pulse to begin an image |
| `uio[1]` | in_valid | in | High on each cycle that carries a weight |
| `uio[7:4]` | weight | in | Signed 4-bit weight, two's complement |
| `uo[3:0]` | digit | out | Predicted digit |
| `uo[4]` | done | out | High when the digit is ready, held until the next `start` |
| `uo[5]` | ready | out | High when the chip accepts an element this cycle |
| `uo[7:6]` | layer | out | Layer being fed (1 to 3), 0 when no image is running |

1. Pulse `start` for one clock, with `in_valid` low.
2. Stream the 1,960 layer-1 weights with the matching pixels, sending only on cycles where `ready` is high.
3. Stream the 800 layer-2 weights (`ready` drops for one cycle before them).
4. Stream the 200 layer-3 weights (`ready` drops for one cycle before them).
5. `done` goes high and `digit` holds the prediction.

The full pin description, host pseudocode and data preparation are in [docs/info.md](docs/info.md).

## Simulation waveforms

These are captures from the RTL simulation (Icarus Verilog) of the first MNIST test image, a 7, driven through the chip pins at a 25 ns clock.
The chip is reset, `start` is pulsed, and the host starts streaming weights and pixels:

![Reset, start pulse and the first weights](./docs/gtkwave_start.png)

The whole inference at a glance: layer 1 takes 1,960 elements, layer 2 takes 800 and layer 3 takes 200 (2,963 cycles, about 74 µs in total):

![Whole inference](./docs/gtkwave_overview.png)

The transition from layer 1 to layer 2: after the last layer-1 weight, `ready` and `in_valid` drop for one cycle while the last neuron is stored, then `layer` becomes 2:

![Layer 1 to layer 2 transition](./docs/gtkwave_boundary.png)

The end of the run: the last layer-3 weights, then `done` goes high and `digit` shows 7:

![Result: done and the predicted digit](./docs/gtkwave_result.png)

## Repository layout

| Path | Contents |
|---|---|
| `src/project.v` | Tiny Tapeout wrapper `tt_um_kaursome10_run_nn`: maps the pins to the network |
| `src/nn_core.v` | The neural network: control, shift-add MAC, requantization, argmax |
| `src/weights/` | Trained int4 weights (`weights_W1/W2/W3.mem`, one hex digit per line) |
| `train/train_qat.py` | Quantization-aware training; exports the `.mem` weight files |
| `tools/eval_accuracy.py` | Bit-exact Python model; reports accuracy on the 10,000 test images |
| `tools/full_accuracy.py` | Runs all 10,000 test images through the RTL with Icarus Verilog |
| `test/` | cocotb testbench (`test.py`, `tb.v`, `Makefile`, `requirements.txt`) and `bulk/tb_bulk.v` for the full 10,000-image run |
| `.github/workflows/full_accuracy.yaml` | Workflow that runs the 10,000-image RTL test and prints the accuracy |
| `docs/` | Datasheet (`info.md`) and the images used in the docs |
| `info.yaml` | Tiny Tapeout project description and pinout |

## Training and evaluating the weights

The training model mirrors the hardware: 7×7 average-pooled pixels, signed int4 weights, ReLU, rounded shifts of 7 and 4, and a 0…63 clamp.
Quantization-aware training uses a straight-through estimator, so the forward pass is bit-exact with the chip while the
gradients flow through the rounding.

```sh
python train/train_qat.py --epochs 300 --h1 40 --h2 20 --A 6 --P 8 --aug 1 --T 32 --lr 0.03 --s1 7 --s2 4 --seed 1 --out src/weights
python tools/eval_accuracy.py                 # prints accuracy on the 10,000 test images
```

`train_qat.py` reads the four MNIST `.gz` files (`train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz`) from the folder in the `MNIST_DIR` environment variable, or from `./mnist/` if it is not set. `tools/eval_accuracy.py` and `tools/full_accuracy.py` download MNIST through torchvision unless you pass `--mnist-dir DIR` with a folder holding those files.

## Testbench

The cocotb testbench drives the design through the Tiny Tapeout wrapper exactly as a real host would: it resets the chip, pulses `start`,
streams the weights and pixels, waits for `done` and compares the result with a bit-exact Python model. Its five tests are:

- `test_reference_model_ranges`: checks that no layer sum overflows the 18-bit accumulator, including for an all-255 image
- `test_images_back_to_back`: several images in a row
- `test_real_mnist_digits`: real MNIST test digits (embedded in `test.py`), 20 samples, of which the chip must classify at least 17 correctly
- `test_random_pauses`: the host pauses at random between weights
- `test_restart_mid_run`: a new `start` in the middle of an image

The same test runs on the gate-level netlist, because it only uses the top-level pins.

```sh
cd test
make                  # RTL simulation
make -B GATES=yes     # gate-level simulation (needs the netlist from the GDS workflow)
```

To measure accuracy on all 10,000 test images, run the **full_accuracy** workflow from the Actions tab (*Run workflow*),
or locally:

```sh
python tools/full_accuracy.py
```

The RTL result, 9,621 of 10,000 images correct, equals the Python model in `tools/eval_accuracy.py` on every image.

![Full 10,000-image RTL run: 96.21 % accuracy](./docs/testbench.png)

How the chip sees the data: the first ten test images, their 7×7 inputs and the digit the RTL predicts (green is correct, red is wrong):

![Example inferences](./docs/inference_examples.png)

## Physical implementation

Hardened with LibreLane (OpenROAD) for SKY130 on a 1×2 Tiny Tapeout tile at a 25 ns clock (40 MHz).

| Metric | Result |
|---|---|
| Core utilization | 78.1 % |
| Cells (Yosys, mapped) | 993 cells, 430 flip-flops, 0 macros |
| Estimated cell area (Yosys) | 16,028 µm² |
| DRC | clean |
| LVS | clean |
| Antenna violations | none |
| Setup and hold timing | met at 25 ns |

After hardening, the layout is DRC and LVS clean. The full 1×2 tile as rendered by the Tiny Tapeout GDS viewer, and a close-up of the metal routing:

![GDS layout of the 1x2 tile](./docs/gds_layout.jpg)

![3D close-up of the routing](./docs/gds_3d.jpg)

## Limitations

- **Not self-contained.** A host controller must stream all 2,960 weights for every inference. Layer 1 also needs the pixel repeated for every row.
- **The host must honour `ready`.** `ready` drops for one cycle after layers 1 and 2, and an element sent on that cycle is lost.
- **High utilization.** The 1×2 tile is about 78 % full, so there is little room for further logic.
- **Accuracy trade-off.** A 7×7 input and 4-bit weights cost accuracy compared with a full-size network. Accuracy was measured on the MNIST test set with 4×4 box-average preprocessing.

## Resources

- [docs/info.md](docs/info.md): datasheet with pins, protocol and test instructions
- [Tiny Tapeout](https://tinytapeout.com) and its [FAQ](https://tinytapeout.com/faq/)
- [Build your design locally](https://www.tinytapeout.com/guides/local-hardening/)
- Enable [GitHub Pages](https://tinytapeout.com/faq/#my-github-action-is-failing-on-the-pages-part) so the GitHub Action can publish the results page.