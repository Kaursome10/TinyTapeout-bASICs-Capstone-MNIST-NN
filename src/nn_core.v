/*
 * Copyright (c) 2024 Simran Kaur
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compact 3-layer MNIST classifier: 49 -> HID1 -> HID2 -> 10.
 *
 * One shared multiply-accumulate unit runs all three layers in order.
 * The host streams one element per accepted cycle, row-major:
 *   layer 1: HID1 rows x 49 cols, each element = (pixel x[k], weight W1[i][k])
 *   layer 2: HID2 rows x HID1 cols, each element = weight W2[i][k]
 *   layer 3: 10 rows x HID2 cols, each element = weight W3[i][k]
 *
 * Each row's sum is turned into its final form one cycle after the row ends:
 *   layers 1-2: ReLU + rounding shift + clamp to 0..2^ABITS-1
 *   layer 3:    compared against the best score so far (running argmax)
 * So the only storage is h1 (HID1 x ABITS) and h2 (HID2 x ABITS).
 *
 * Only the top PBITS bits of the pixel input are used.
 */

`default_nettype none

module nn_core #(
    parameter HID1     = 40,
    parameter HID2     = 20,
    parameter ABITS    = 6,   // bits per stored hidden activation
    parameter PBITS    = 8,   // bits of each pixel actually used (top bits of pixel[7:0])
    parameter R1_SHIFT = 7,
    parameter R2_SHIFT = 4,
    parameter SR       = 1    // 1: hidden activations in rotating shift registers (no read mux)
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,     // begin a new inference (also aborts one in progress)
    input  wire       in_valid,  // an element is present this cycle
    input  wire [7:0] pixel,     // x[k], only used in layer 1
    input  wire [3:0] weight,    // W[i][k], signed int4
    output wire       ready,     // high while elements are being accepted
    output wire [1:0] layer_o,   // layer being fed (1, 2, 3), 0 when not accepting
    output reg        done,      // result is valid; stays high until the next start
    output reg  [3:0] digit      // predicted digit 0-9
);

    localparam AMAX   = (1 << ABITS) - 1;
    localparam MAX1   = 49   * 8 * ((1 << PBITS) - 1);
    localparam MAX2   = HID1 * 8 * AMAX;
    localparam MAX3   = HID2 * 8 * AMAX;
    localparam MAXALL = (MAX1 > MAX2) ? ((MAX1 > MAX3) ? MAX1 : MAX3) : ((MAX2 > MAX3) ? MAX2 : MAX3);
    localparam ACC_W  = $clog2(MAXALL + 1) + 1;
    localparam XW     = ((ABITS > PBITS) ? ABITS : PBITS) + 1;
    localparam H1AW   = $clog2(HID1);
    localparam H2AW   = $clog2(HID2);

    reg        running;
    reg  [1:0] layer;
    reg  [5:0] row;
    reg  [5:0] col;

    wire [5:0] m_last = (layer == 2'd1) ? 6'd48 : (layer == 2'd2) ? (HID1 - 1) : (HID2 - 1);
    wire [5:0] n_last = (layer == 2'd1) ? (HID1 - 1) : (layer == 2'd2) ? (HID2 - 1) : 6'd9;

    reg signed [ACC_W-1:0] acc;

    // A row finished last cycle: its sum is in acc and gets written out this cycle.
    reg        pend;
    reg        pend_last;
    reg  [1:0] pend_layer;
    reg  [5:0] pend_row;

    reg signed [ACC_W-1:0] best;
    reg  [3:0]             best_idx;

    reg [ABITS-1:0] h1 [0:HID1-1];
    reg [ABITS-1:0] h2 [0:HID2-1];

    // With shift-register storage, the cycle after a layer's last row is spent
    // committing that row, so the host must wait for ready.
    wire stall = (SR != 0) && pend && pend_last && (pend_layer != 2'd3);

    assign ready   = running && !stall;
    assign layer_o = running ? layer : 2'd0;

    // ---------------------------------------------------------------
    // Multiply-accumulate
    // ---------------------------------------------------------------
    reg signed [XW-1:0] x;
    always @(*) begin
        case (layer)
            2'd1:    x = {{(XW-PBITS){1'b0}}, pixel[7 -: PBITS]};
            2'd2:    x = {{(XW-ABITS){1'b0}}, (SR != 0) ? h1[0] : h1[col[H1AW-1:0]]};
            default: x = {{(XW-ABITS){1'b0}}, (SR != 0) ? h2[0] : h2[col[H2AW-1:0]]};
        endcase
    end

    wire signed [3:0]       w        = weight;
    wire signed [XW+3:0]    prod     = w * x;
    wire signed [ACC_W-1:0] prod_ext = {{(ACC_W-XW-4){prod[XW+3]}}, prod};
    wire signed [ACC_W-1:0] acc_base = (col == 6'd0) ? {ACC_W{1'b0}} : acc;
    wire signed [ACC_W-1:0] sum      = acc_base + prod_ext;

    wire accept    = running && in_valid && !stall;
    wire row_end   = accept && (col == m_last);

    // ---------------------------------------------------------------
    // ReLU + requantize (applied to acc while pend is high)
    // ---------------------------------------------------------------
    wire [ACC_W-1:0] relu_acc = acc[ACC_W-1] ? {ACC_W{1'b0}} : acc;

    wire [ACC_W:0] round1 = {{ACC_W{1'b0}}, 1'b1} << (R1_SHIFT - 1);
    wire [ACC_W:0] round2 = {{ACC_W{1'b0}}, 1'b1} << (R2_SHIFT - 1);
    wire [ACC_W:0] shr1   = ({1'b0, relu_acc} + round1) >> R1_SHIFT;
    wire [ACC_W:0] shr2   = ({1'b0, relu_acc} + round2) >> R2_SHIFT;
    wire [ABITS-1:0] q1   = (|shr1[ACC_W:ABITS]) ? AMAX[ABITS-1:0] : shr1[ABITS-1:0];
    wire [ABITS-1:0] q2   = (|shr2[ACC_W:ABITS]) ? AMAX[ABITS-1:0] : shr2[ABITS-1:0];

    wire better = (pend_row == 6'd0) || (acc > best);

    // ---------------------------------------------------------------
    // Control
    // ---------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running    <= 1'b0;
            layer      <= 2'd1;
            row        <= 6'd0;
            col        <= 6'd0;
            acc        <= {ACC_W{1'b0}};
            pend       <= 1'b0;
            pend_last  <= 1'b0;
            pend_layer <= 2'd0;
            pend_row   <= 6'd0;
            best       <= {ACC_W{1'b0}};
            best_idx   <= 4'd0;
            done       <= 1'b0;
            digit      <= 4'd0;
        end else begin
            pend <= 1'b0;

            if (start) begin
                running <= 1'b1;
                layer   <= 2'd1;
                row     <= 6'd0;
                col     <= 6'd0;
                done    <= 1'b0;
            end else begin
                if (accept) begin
                    acc <= sum;
                    if (row_end) begin
                        col        <= 6'd0;
                        pend       <= 1'b1;
                        pend_layer <= layer;
                        pend_row   <= row;
                        pend_last  <= (row == n_last);
                        if (row == n_last) begin
                            row <= 6'd0;
                            if (layer == 2'd3)
                                running <= 1'b0;
                            else
                                layer <= layer + 2'd1;
                        end else begin
                            row <= row + 6'd1;
                        end
                    end else begin
                        col <= col + 6'd1;
                    end
                end

                if (pend && pend_layer == 2'd3) begin
                    if (better) begin
                        best     <= acc;
                        best_idx <= pend_row[3:0];
                    end
                    if (pend_row == 6'd9) begin
                        done  <= 1'b1;
                        digit <= better ? pend_row[3:0] : best_idx;
                    end
                end
            end
        end
    end

    // Hidden activation storage (no reset needed: always written before read)
    integer i;
    generate
        if (SR == 0) begin : g_addr
            always @(posedge clk) begin
                if (pend && pend_layer == 2'd1) h1[pend_row[H1AW-1:0]] <= q1;
                if (pend && pend_layer == 2'd2) h2[pend_row[H2AW-1:0]] <= q2;
            end
        end else begin : g_ring
            wire wr1  = pend && pend_layer == 2'd1;
            wire rot1 = accept && layer == 2'd2;
            wire wr2  = pend && pend_layer == 2'd2;
            wire rot2 = accept && layer == 2'd3;
            always @(posedge clk) begin
                if (wr1 || rot1) begin
                    for (i = 0; i < HID1 - 1; i = i + 1) h1[i] <= h1[i+1];
                    h1[HID1-1] <= wr1 ? q1 : h1[0];
                end
                if (wr2 || rot2) begin
                    for (i = 0; i < HID2 - 1; i = i + 1) h2[i] <= h2[i+1];
                    h2[HID2-1] <= wr2 ? q2 : h2[0];
                end
            end
        end
    endgenerate

endmodule