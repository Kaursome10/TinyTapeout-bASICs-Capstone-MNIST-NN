/*
 * Copyright (c) 2024 Simran Kaur
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compact 3-layer MNIST classifier: 49 -> 50 -> 25 -> 10.
 *
 * One shared multiply-accumulate unit runs all three layers in order.
 * The host streams one element per accepted cycle, row-major:
 *   layer 1: 50 rows x 49 cols, each element = (pixel x[k], weight W1[i][k])
 *   layer 2: 25 rows x 50 cols, each element = weight W2[i][k]
 *   layer 3: 10 rows x 25 cols, each element = weight W3[i][k]
 * 2450 + 1250 + 250 = 3950 elements per image.
 *
 * Each row's sum is turned into its final form one cycle after the row ends:
 *   layers 1-2: ReLU + rounding shift + clamp to 0..127, stored as 7 bits
 *   layer 3:    compared against the best score so far (running argmax)
 * So the only storage is h1 (50 x 7 bits) and h2 (25 x 7 bits).
 *
 * The math matches the original run_NN design bit for bit, except for the
 * requant shifts (see R1_SHIFT / R2_SHIFT) and unsigned pixels (PIXEL_SIGNED).
 */

`default_nettype none

module nn_core #(
    parameter ACC_W        = 18,  // worst case |sum| ~ 1e5, fits in 18-bit signed
    // Requant shifts. The original 10 / 9 were set for the old int8 weights;
    // the int4 weights are ~11-13x smaller, so 10 / 9 rounds every layer-2
    // value to 0. 6 / 5 restores the int8 model's scale.
    parameter R1_SHIFT     = 6,   // layer 1 requant: divide by 2^6
    parameter R2_SHIFT     = 5,   // layer 2 requant: divide by 2^5
    parameter PIXEL_SIGNED = 0    // 0: pixels are uint8 (0..255), each the average of a 4x4
                                  //    block of the 28x28 MNIST image (91.2% on MNIST test set)
                                  // 1: pixels are int8 (-128..127), as in the original design
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

    reg        running;
    reg  [1:0] layer;
    reg  [5:0] row;
    reg  [5:0] col;

    // Last column / last row index for each layer
    wire [5:0] m_last = (layer == 2'd1) ? 6'd48 : (layer == 2'd2) ? 6'd49 : 6'd24;
    wire [5:0] n_last = (layer == 2'd1) ? 6'd49 : (layer == 2'd2) ? 6'd24 : 6'd9;

    reg signed [ACC_W-1:0] acc;

    // A row finished last cycle: its sum is in acc and gets written out this cycle.
    reg        pend;
    reg  [1:0] pend_layer;
    reg  [5:0] pend_row;

    // Running argmax over the layer 3 scores
    reg signed [ACC_W-1:0] best;
    reg  [3:0]             best_idx;

    // Hidden activations (always 0..127, so 7 bits each)
    reg [6:0] h1 [0:49];
    reg [6:0] h2 [0:24];

    assign ready   = running;
    assign layer_o = running ? layer : 2'd0;

    // ---------------------------------------------------------------
    // Multiply-accumulate
    // ---------------------------------------------------------------
    reg signed [8:0] x;
    always @(*) begin
        case (layer)
            2'd1:    x = PIXEL_SIGNED ? {pixel[7], pixel} : {1'b0, pixel};
            2'd2:    x = {2'b00, h1[col]};
            default: x = {2'b00, h2[col[4:0]]};
        endcase
    end

    wire signed [3:0]       w        = weight;
    wire signed [12:0]      prod     = w * x;
    wire signed [ACC_W-1:0] prod_ext = {{(ACC_W-13){prod[12]}}, prod};
    wire signed [ACC_W-1:0] acc_base = (col == 6'd0) ? {ACC_W{1'b0}} : acc;
    wire signed [ACC_W-1:0] sum      = acc_base + prod_ext;

    wire accept    = running && in_valid;
    wire row_end   = accept && (col == m_last);

    // ---------------------------------------------------------------
    // ReLU + requantize (applied to acc while pend is high)
    // ---------------------------------------------------------------
    wire [ACC_W-1:0] relu_acc = acc[ACC_W-1] ? {ACC_W{1'b0}} : acc;

    wire [ACC_W:0] round1 = {{ACC_W{1'b0}}, 1'b1} << (R1_SHIFT - 1);
    wire [ACC_W:0] round2 = {{ACC_W{1'b0}}, 1'b1} << (R2_SHIFT - 1);
    wire [ACC_W:0] shr1   = ({1'b0, relu_acc} + round1) >> R1_SHIFT;
    wire [ACC_W:0] shr2   = ({1'b0, relu_acc} + round2) >> R2_SHIFT;
    wire [6:0]     q1     = (|shr1[ACC_W:7]) ? 7'd127 : shr1[6:0];
    wire [6:0]     q2     = (|shr2[ACC_W:7]) ? 7'd127 : shr2[6:0];

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
    always @(posedge clk) begin
        if (pend && pend_layer == 2'd1) h1[pend_row]      <= q1;
        if (pend && pend_layer == 2'd2) h2[pend_row[4:0]] <= q2;
    end

endmodule
