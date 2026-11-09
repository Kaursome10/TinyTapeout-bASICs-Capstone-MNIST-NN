# Testbench

This folder holds the [cocotb](https://docs.cocotb.org/en/stable/) testbench for `tt_um_kaursome10_run_nn`
(see the [Tiny Tapeout testing guide](https://tinytapeout.com/hdl/testing/) for background). The testbench drives
only the top-level pins, the way a real host would, and compares every result with a bit-exact Python model.

| File | Purpose |
|---|---|
| `test.py` | The five cocotb tests and the reference model |
| `tb.v` | Verilog wrapper that instantiates the design and dumps `tb.fst` |
| `Makefile` | Builds and runs the simulation (RTL or gate level) |
| `requirements.txt` | Python packages (cocotb, pytest) |
| `mnist_test_images.mem`, `mnist_test_labels.mem` | The 10,000 MNIST test images, average-pooled to 7×7, and their labels |
| `t10k-images-idx3-ubyte.gz`, `t10k-labels-idx1-ubyte.gz` | The raw MNIST test set that `generate.py` reads |
| `generate.py` | Rebuilds the two `.mem` files from the raw MNIST files (numpy only) |
| `tb.gtkw` | GTKWave signal layout for `tb.fst` |

## Running

You need Icarus Verilog (`iverilog`) and the Python packages:

```sh
pip install -r requirements.txt
```

RTL simulation:

```sh
make -B
python -m cocotb_tools.check_results results.xml   # make succeeds even if a test fails, so check the results
```

Gate-level simulation: first harden the project (the GDS workflow does this and produces the netlist), copy the netlist
to `gate_level_netlist.v` in this folder, set `PDK_ROOT` to your SKY130 PDK, and run:

```sh
make -B GATES=yes
```

To save the waveform as VCD instead of FST, edit `tb.v` to use `$dumpfile("tb.vcd");` and run `make -B FST=`.

## Viewing waveforms

```sh
gtkwave tb.fst tb.gtkw
# or
surfer tb.fst
```

## All 10,000 images

These tests run only a few dozen images. To run every MNIST test image through the Verilog, use
`python ../tools/full_accuracy.py` (see the [top-level README](../README.md)).
