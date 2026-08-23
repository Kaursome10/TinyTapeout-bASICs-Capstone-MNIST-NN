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

    wire        start   = uio_in[0];
    wire        ld_en   = uio_in[1];
    wire [5:0]  ld_addr = uio_in[7:2];
    wire [7:0]  ld_data = ui_in;

    wire        valid_o;
    wire [3:0]  out_o;

    assign uo_out  = {3'b000, valid_o, out_o};
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    // List all unused inputs to prevent warnings
    wire _unused = &{ena, 1'b0};

    run_NN nn (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (start),
        .ld_en   (ld_en),
        .ld_addr (ld_addr),
        .ld_data (ld_data),
        .valid   (valid_o),
        .out     (out_o)
    );

endmodule