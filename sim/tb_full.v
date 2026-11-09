`timescale 1ns / 1ps
// Full-chip MNIST run for tt_um_kaursome10_run_nn.
//
// Streams the weights and the 7x7 pixels of every test image through the real
// Tiny Tapeout pins, the same way test/test.py does (inputs change on the falling
// edge, `ready` is honoured), and writes the predicted digit of each image to a file.
// It is meant to be driven by tools/full_accuracy.py, which passes the file names:
//
//   +W1=<file> +W2=<file> +W3=<file>   weight files, one hex digit per line
//   +PIX=<file>                        pooled 7x7 images, 49 hex bytes per image
//   +OUT=<file>                        predicted digit per image (written)
//   +N=<images>                        number of images to run (default 10000)
module tb_full;
  localparam MAX_IMG = 10000;
  localparam W1_N = 1960;
  localparam W2_N = 800;
  localparam W3_N = 200;
  localparam NEL  = W1_N + W2_N + W3_N;      // elements per image

  reg clk = 0;
  reg rst_n = 0;
  reg ena = 1;
  reg [7:0] ui_in = 0;
  reg [7:0] uio_in = 0;
  wire [7:0] uo_out, uio_out, uio_oe;

  tt_um_kaursome10_run_nn dut (
      .ui_in(ui_in), .uo_out(uo_out), .uio_in(uio_in),
      .uio_out(uio_out), .uio_oe(uio_oe), .ena(ena), .clk(clk), .rst_n(rst_n));

  always #5 clk = ~clk;

  reg [3:0] wmem [0:NEL-1];
  reg [7:0] pmem [0:MAX_IMG*49-1];

  wire       ready = uo_out[5];
  wire       done  = uo_out[4];
  wire [3:0] digit = uo_out[3:0];

  integer n_img, img, idx, waited, fout;
  integer stall_errors, done_timeouts;
  reg [1023:0] w1_file, w2_file, w3_file, pix_file, out_file;

  initial begin
    if (!$value$plusargs("W1=%s", w1_file) || !$value$plusargs("W2=%s", w2_file) ||
        !$value$plusargs("W3=%s", w3_file) || !$value$plusargs("PIX=%s", pix_file) ||
        !$value$plusargs("OUT=%s", out_file)) begin
      $display("ERROR: need +W1= +W2= +W3= +PIX= +OUT=");
      $finish;
    end
    if (!$value$plusargs("N=%d", n_img)) n_img = MAX_IMG;
    if (n_img > MAX_IMG) n_img = MAX_IMG;

    $readmemh(w1_file, wmem, 0, W1_N - 1);
    $readmemh(w2_file, wmem, W1_N, W1_N + W2_N - 1);
    $readmemh(w3_file, wmem, W1_N + W2_N, NEL - 1);
    $readmemh(pix_file, pmem);

    fout = $fopen(out_file, "w");
    stall_errors = 0;
    done_timeouts = 0;

    repeat (10) @(posedge clk);
    rst_n = 1;
    @(negedge clk);

    for (img = 0; img < n_img; img = img + 1) begin
      // One-cycle start pulse with in_valid low.
      ui_in  = 0;
      uio_in = 8'b0000_0001;
      @(negedge clk);
      uio_in = 0;

      for (idx = 0; idx < NEL; idx = idx + 1) begin
        // `ready` may drop for one cycle at each layer boundary (idx 1960 and 2760).
        waited = 0;
        while (!ready) begin
          if (!(idx == W1_N || idx == W1_N + W2_N) || waited >= 2) stall_errors = stall_errors + 1;
          uio_in = 0;
          @(negedge clk);
          waited = waited + 1;
          if (waited > 5) begin
            $display("ERROR: chip stuck not ready, image %0d element %0d", img, idx);
            $finish;
          end
        end
        // Layer 1 needs the matching pixel with every weight; later layers ignore ui_in.
        ui_in  = (idx < W1_N) ? pmem[img * 49 + (idx % 49)] : 8'd0;
        uio_in = {wmem[idx], 2'b00, 1'b1, 1'b0};   // weight, in_valid, start
        @(negedge clk);
      end
      ui_in  = 0;
      uio_in = 0;

      waited = 0;
      while (!done && waited < 10) begin
        @(negedge clk);
        waited = waited + 1;
      end
      if (!done) done_timeouts = done_timeouts + 1;
      $fdisplay(fout, "%0d", digit);
    end

    $fclose(fout);
    $display("DONE images=%0d stall_errors=%0d done_timeouts=%0d", n_img, stall_errors, done_timeouts);
    $finish;
  end
endmodule
