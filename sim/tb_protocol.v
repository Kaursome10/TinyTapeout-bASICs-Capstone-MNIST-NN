`timescale 1ns/1ps
// Protocol-level Icarus Verilog testbench for tt_um_kaursome10_run_nn (drives only the Tiny Tapeout pins).
// Run from the project root:
//   iverilog -g2012 -o sim.vvp src/run_NN.sv src/project.sv sim/tb_protocol.v
//   vvp sim.vvp +PIX=sim/pix.mem +N=10 +GAP=2 [+PAUSE=30] [+VCD]
// pix.mem: one pooled 7x7 image per 49 lines, one hex byte per line.
//   +N=<images> +GAP=<idle cycles between layers> +PAUSE=<% random data_valid=0 cycles inside a layer>
//   +VCD  -> dump waveform of the first image
module tb_protocol;
  reg clk = 0, rst_n = 0, ena = 1;
  reg [7:0] ui_in = 0, uio_in = 0;
  wire [7:0] uo_out, uio_out, uio_oe;

  tt_um_kaursome10_run_nn dut (.ui_in(ui_in), .uo_out(uo_out), .uio_in(uio_in),
      .uio_out(uio_out), .uio_oe(uio_oe), .ena(ena), .clk(clk), .rst_n(rst_n));

  wire       start      = uio_in[0];
  wire       data_valid = uio_in[1];
  wire [3:0] weight     = uio_in[7:4];
  wire       valid      = uo_out[4];
  wire [3:0] digit      = uo_out[3:0];

  always #26 clk = ~clk;                       // 52 ns period, 19.23 MHz

  reg [3:0] W1 [0:2449];
  reg [3:0] W2 [0:1249];
  reg [3:0] W3 [0:249];
  reg [7:0] PIX [0:489999];

  integer N, GAP, PAUSE, seed, img, r, c, i, lat, hold_ok, single_ok, wcycles, total_cycles, tc0;
  integer k;
  reg got;
  reg [1023:0] pixfile;

  // one weight per clock; optional random idle cycles (data_valid low) before it
  task send(input [7:0] pix, input [3:0] w);
    begin
      while (PAUSE > 0 && ($urandom(seed) % 100) < PAUSE) begin
        @(negedge clk); uio_in = 8'h00;
      end
      @(negedge clk);
      ui_in  = pix;
      uio_in = {w, 2'b00, 1'b1, 1'b0};          // weight[7:4], data_valid=uio[1], start=uio[0]
      wcycles = wcycles + 1;
    end
  endtask

  task idle(input integer n);
    integer j;
    begin
      for (j = 0; j < n; j = j + 1) begin @(negedge clk); uio_in = 8'h00; end
    end
  endtask

  initial begin
    if (!$value$plusargs("N=%d", N)) N = 1;
    if (!$value$plusargs("GAP=%d", GAP)) GAP = 2;
    if (!$value$plusargs("PAUSE=%d", PAUSE)) PAUSE = 0;
    seed = 12345;
    $readmemh("src/weights/weights_W1.mem", W1);
    $readmemh("src/weights/weights_W2.mem", W2);
    $readmemh("src/weights/weights_W3.mem", W3);
    if (!$value$plusargs("PIX=%s", pixfile)) pixfile = "sim/pix.mem";
    $readmemh(pixfile, PIX);
    if ($test$plusargs("VCD")) begin
      $dumpfile("run.vcd");
      $dumpvars(0, tb_protocol.clk, tb_protocol.rst_n, tb_protocol.start, tb_protocol.data_valid, tb_protocol.ui_in, tb_protocol.weight, tb_protocol.valid, tb_protocol.digit);
      $dumpvars(0, dut.nn.layer, dut.nn.row, dut.nn.col, dut.nn.acc, dut.nn.running, dut.nn.pending);
    end

    repeat (5) @(negedge clk);
    rst_n = 1;
    repeat (3) @(negedge clk);

    for (img = 0; img < N; img = img + 1) begin
      wcycles = 0;
      @(negedge clk); uio_in = 8'h01;           // start pulse, data_valid low
      @(negedge clk); uio_in = 8'h00;
      for (r = 0; r < 50; r = r + 1) for (c = 0; c < 49; c = c + 1) send(PIX[img*49 + c], W1[r*49 + c]);
      idle(GAP);
      for (i = 0; i < 1250; i = i + 1) send(8'h00, W2[i]);
      idle(GAP);
      for (i = 0; i < 250; i = i + 1) send(8'h00, W3[i]);
      // last weight is accepted at the next posedge; count edges until valid is seen
      lat = 0; got = 0;
      @(posedge clk);
      @(negedge clk); uio_in = 8'h00;
      for (k = 0; k < 40 && !got; k = k + 1) begin
        @(posedge clk); #1;
        if (uo_out[4]) begin got = 1; lat = k + 1; end
      end
      if (!got) begin
        $display("IMG %0d NOVALID", img);
      end else begin
        // valid must be a one-cycle pulse, and out must hold
        r = uo_out[3:0];
        @(posedge clk); #1; single_ok = (uo_out[4] == 0);
        repeat (20) @(posedge clk);
        #1; hold_ok = (uo_out[3:0] == r);
        if ($test$plusargs("DBG")) begin for (i=0;i<50;i=i+1) $write("%0d ", dut.nn.h1[i]); $write("\nH2: "); for (i=0;i<25;i=i+1) $write("%0d ", dut.nn.h2[i]); $write("\n"); end
        $display("IMG %0d PRED %0d LAT %0d WEIGHTCYCLES %0d PULSE1 %0d HOLD %0d", img, r, lat, wcycles, single_ok, hold_ok);
      end
    end
    $finish;
  end
endmodule
