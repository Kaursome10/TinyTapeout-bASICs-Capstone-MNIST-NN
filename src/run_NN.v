`timescale 1ns/1ps

module run_NN #(
    parameter DW      = 8,
    parameter IMG_SIZE = 7 * 7, // 49

    // Power-of-two requant scale factors -- see requant.sv. Nearest
    // power-of-two approximation to your calibrated scales; re-derive
    // from your own calibration script if you retrain.
    parameter R1_SHIFT = 5'd10,
    parameter R2_SHIFT = 5'd9
) (
  input clk,
  input rst_n,
  input start,

  // vector image interface (load the 49 pixels before pulsing start)
  input  wire ld_en,
  input  wire [$clog2(IMG_SIZE)-1:0] ld_addr,
  input  wire signed [DW-1:0]   ld_data,

  // streamed weight interface: one 8-bit weight per pulse, in the same
  // row-major order the old .mem files were already written in --
  // layer 1's N1*M1 weights, then layer 2's N2*M2, then layer 3's N3*M3.
  // Broadcast to all three layers: only the layer currently in COMPUTE
  // state will actually respond to a given pulse, so no muxing is needed.
  input  wire                   weight_valid,
  input  wire signed [DW-1:0]   weight_data,

  // classified number
  output wire valid,
  output wire [3:0] out
);

    localparam N1 = 50,  M1 = IMG_SIZE; // layer1: 49 -> 64
    localparam N2 = 25,  M2 = 50;       // layer2: 64 -> 32
    localparam N3 = 10,  M3 = 25;       // layer3: 32 -> 10

    localparam MAXN = 50; // widest layer, sizes the drain counter

    // ---------------------------------------------------------------
    // FSM
    // ---------------------------------------------------------------
    localparam S_IDLE      = 0,
               S_H1_RUN    = 1,
               S_H1_DRAIN  = 2,
               S_H2_RUN    = 3,
               S_H2_DRAIN  = 4,
               S_H3_RUN    = 5,
               S_H3_DRAIN  = 6,
               S_DONE      = 7;

    reg [2:0] state;
    reg [$clog2(MAXN)-1:0] cnt;

    reg h1_start, h2_start, h3_start;
    wire h1_done, h2_done, h3_done;

    // ld_en shares a physical pin with weight_valid (both driven by the
    // same top-level "data_valid" wire). weight_valid pulses continuously
    // during compute, so ld_en must be explicitly gated to S_IDLE only --
    // otherwise every weight-streaming pulse would also spuriously
    // rewrite h1's pixel vector memory mid-computation.
    wire h1_pixel_ld_en = ld_en && (state == S_IDLE);

    // ---------------------------------------------------------------
    // Layer 1: image(49) x W1(64x49) -> h1out(64)
    // ---------------------------------------------------------------
    wire signed [31:0] h1out;
    wire signed [31:0] h1out_relu;
    wire [DW-1:0]      h1out_q;

    wire [$clog2(N1)-1:0] h1_rd_addr;
    assign h1_rd_addr = cnt[$clog2(N1)-1:0];

    matvec_top_banked #(.N(N1), .M(M1), .DW(DW), .ACC_W(32)) h1 (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (h1_start),
        .done         (h1_done),
        .ld_en        (h1_pixel_ld_en), // image pixels loaded straight through into h1's vector mem, gated to idle-only
        .ld_addr      (ld_addr),
        .ld_data      (ld_data),
        .weight_valid (weight_valid),
        .weight_data  (weight_data),
        .rd_en        (state == S_H1_DRAIN),
        .rd_addr      (h1_rd_addr),
        .rd_data      (h1out)
    );

    relu #(.W(32)) relu1 (.in(h1out), .out(h1out_relu));
    requant #(.SHIFT(R1_SHIFT)) rq1 (.acc(h1out_relu), .h_out(h1out_q));

    // ---------------------------------------------------------------
    // Layer 2: h1out_q(64) x W2(32x64) -> h2out(32)
    // ---------------------------------------------------------------
    wire signed [31:0] h2out;
    wire signed [31:0] h2out_relu;
    wire [DW-1:0]      h2out_q;

    wire [$clog2(N2)-1:0] h2_rd_addr;
    assign h2_rd_addr = cnt[$clog2(N2)-1:0];

    wire h1_drain_wr_en = (state == S_H1_DRAIN) && (cnt < N1);

    matvec_top_banked #(.N(N2), .M(M2), .DW(DW), .ACC_W(32)) h2 (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (h2_start),
        .done         (h2_done),
        .ld_en        (h1_drain_wr_en),
        .ld_addr      (cnt[$clog2(M2)-1:0]),
        .ld_data      (h1out_q),
        .weight_valid (weight_valid),
        .weight_data  (weight_data),
        .rd_en        (state == S_H2_DRAIN),
        .rd_addr      (h2_rd_addr),
        .rd_data      (h2out)
    );

    relu #(.W(32)) relu2 (.in(h2out), .out(h2out_relu));
    requant #(.SHIFT(R2_SHIFT)) rq2 (.acc(h2out_relu), .h_out(h2out_q));

    // ---------------------------------------------------------------
    // Layer 3: h2out_q(32) x W3(10x32) -> final(10)  (no ReLU/requant)
    // ---------------------------------------------------------------
    wire signed [31:0] final_out;

    wire [$clog2(N3)-1:0] h3_rd_addr;
    assign h3_rd_addr = cnt[$clog2(N3)-1:0];

    wire h2_drain_wr_en = (state == S_H2_DRAIN) && (cnt < N2);

    matvec_top_banked #(.N(N3), .M(M3), .DW(DW), .ACC_W(32)) h3 (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (h3_start),
        .done         (h3_done),
        .ld_en        (h2_drain_wr_en),
        .ld_addr      (cnt[$clog2(M3)-1:0]),
        .ld_data      (h2out_q),
        .weight_valid (weight_valid),
        .weight_data  (weight_data),
        .rd_en        (state == S_H3_DRAIN),
        .rd_addr      (h3_rd_addr),
        .rd_data      (final_out)
    );

    // ---------------------------------------------------------------
    // Collect final layer's 10 scores, then argmax
    // ---------------------------------------------------------------
    reg signed [319:0] scores_flat;
    wire [3:0] argmax_idx;

    always @(posedge clk) begin
        if (state == S_H3_DRAIN && cnt < N3) begin
            case (cnt[3:0])
                4'd0: scores_flat[31:0]     <= final_out;
                4'd1: scores_flat[63:32]    <= final_out;
                4'd2: scores_flat[95:64]    <= final_out;
                4'd3: scores_flat[127:96]  <= final_out;
                4'd4: scores_flat[159:128]  <= final_out;
                4'd5: scores_flat[191:160]  <= final_out;
                4'd6: scores_flat[223:192]  <= final_out;
                4'd7: scores_flat[255:224]  <= final_out;
                4'd8: scores_flat[287:256]  <= final_out;
                4'd9: scores_flat[319:288]  <= final_out;
                default: ; // ignore out-of-range cnt values
            endcase
        end
    end

    argmax amax (.in_flat(scores_flat), .max_index(argmax_idx));

    reg valid_r;
    reg [3:0] out_r;
    assign valid = valid_r;
    assign out   = out_r;

    // ---------------------------------------------------------------
    // Main FSM
    // ---------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            cnt      <= 0;
            h1_start <= 1'b0;
            h2_start <= 1'b0;
            h3_start <= 1'b0;
            valid_r  <= 1'b0;
            out_r    <= 4'd0;
        end
        else begin
            h1_start <= 1'b0;
            h2_start <= 1'b0;
            h3_start <= 1'b0;

            case (state)
                S_IDLE: begin
                    valid_r <= 1'b0;
                    if (start) begin
                        h1_start <= 1'b1;
                        state    <= S_H1_RUN;
                    end
                end

                S_H1_RUN: begin
                    if (h1_done) begin
                        cnt   <= 0;
                        state <= S_H1_DRAIN;
                    end
                end

                S_H1_DRAIN: begin
                    cnt <= cnt + 1;
                    if (cnt == N1 - 1) begin
                        h2_start <= 1'b1;
                        state    <= S_H2_RUN;
                    end
                end

                S_H2_RUN: begin
                    if (h2_done) begin
                        cnt   <= 0;
                        state <= S_H2_DRAIN;
                    end
                end

                S_H2_DRAIN: begin
                    cnt <= cnt + 1;
                    if (cnt == N2 - 1) begin
                        h3_start <= 1'b1;
                        state    <= S_H3_RUN;
                    end
                end

                S_H3_RUN: begin
                    if (h3_done) begin
                        cnt   <= 0;
                        state <= S_H3_DRAIN;
                    end
                end

                S_H3_DRAIN: begin
                    cnt <= cnt + 1;
                    if (cnt == N3 - 1) begin
                        state <= S_DONE;
                    end
                end

                S_DONE: begin
                    valid_r <= 1'b1;
                    out_r   <= argmax_idx;
                    state   <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule