/*
 * Copyright (c) 2024 Simran Kaur
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_kaursome10_run_nn (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

    // Inputs
    //   ui_in[7:0]  pixel x[k] (layer 1 only)
    //   uio_in[0]   start
    //   uio_in[1]   in_valid: an element is present this cycle
    //   uio_in[7:4] weight W[i][k] (signed int4)
    // Outputs
    //   uo_out[3:0] predicted digit
    //   uo_out[4]   done (held until the next start)
    //   uo_out[5]   ready (elements are being accepted)
    //   uo_out[7:6] layer being fed (1-3), 0 when not accepting

    wire       ready;
    wire       done;
    wire [1:0] layer;
    wire [3:0] digit;

    nn_core nn (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (uio_in[0]),
        .in_valid (uio_in[1]),
        .pixel    (ui_in),
        .weight   (uio_in[7:4]),
        .ready    (ready),
        .layer_o  (layer),
        .done     (done),
        .digit    (digit)
    );

    assign uo_out  = {layer, ready, done, digit};
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;  // all uio pins are inputs

    // List all unused inputs to prevent warnings
    wire _unused = &{ena, uio_in[3:2], 1'b0};

endmodule
