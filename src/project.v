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

    // uio_in[1] ("data_valid") is reused for two different meanings
    // depending on phase, since pixel loading and weight streaming never
    // happen at the same time:
    //   - before `start`: it's ld_en, gating pixel writes (paired with
    //     ld_addr on uio_in[7:2] and pixel byte on ui_in)
    //   - during compute (after `start`, before `valid`): it's
    //     weight_valid, gating one streamed weight byte per pulse on
    //     ui_in. ld_addr is ignored during this phase since weight order
    //     is implicit (matches the row-major order weights were already
    //     stored in), so no addressing is needed.
    wire        start        = uio_in[0];
    wire        data_valid   = uio_in[1];
    wire [5:0]  ld_addr      = uio_in[7:2];
    wire [7:0]  data_in      = ui_in;

    wire        valid_o;
    wire [3:0]  out_o;

    assign uo_out  = {3'b000, valid_o, out_o};
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    // List all unused inputs to prevent warnings
    wire _unused = &{ena, 1'b0};

    run_NN nn (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .ld_en        (data_valid),
        .ld_addr      (ld_addr),
        .ld_data      (data_in),
        .weight_valid (data_valid),
        .weight_data  (data_in),
        .valid        (valid_o),
        .out          (out_o)
    );

endmodule