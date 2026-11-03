`timescale 1ns/1ps

module run_NN #(
    parameter DW = 8,
    parameter WW = 4,
    parameter IMG_SIZE = 7 * 7,
    parameter R1_SHIFT = 5'd6,
    parameter R2_SHIFT = 5'd5,
    parameter ACC_W = 18
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire ld_en,
    input wire [$clog2(IMG_SIZE)-1:0] ld_addr,
    input wire [DW-1:0] ld_data,
    input wire weight_valid,
    input wire signed [WW-1:0] weight_data,
    output wire valid,
    output wire [3:0] out
);
    localparam L1 = 1;
    localparam L2 = 2;
    localparam L3 = 3;

    (* keep *) reg running;
    (* keep *) reg [1:0] layer;
    (* keep *) reg [5:0] row;
    (* keep *) reg [5:0] col;
    (* keep *) reg [5:0] pending_row;
    (* keep *) reg [1:0] pending_layer;
    (* keep *) reg pending;

    reg [6:0] h1 [0:49];
    reg [6:0] h2 [0:24];
    reg signed [ACC_W-1:0] acc;
    reg signed [ACC_W-1:0] best_score;
    reg [3:0] best_index;
    reg valid_r;
    reg [3:0] out_r;
    wire _unused = &{ld_en, ld_addr};

    wire [5:0] last_col = (layer == L1) ? 6'd48 :
                          (layer == L2) ? 6'd49 : 6'd24;
    wire [5:0] last_row = (layer == L1) ? 6'd49 :
                          (layer == L2) ? 6'd24 : 6'd9;

    reg [DW-1:0] activation;
    always @(*) begin
        case (layer)
            L1: activation = ld_data;
            L2: activation = {1'b0, h1[col]};
            default: activation = {1'b0, h2[col[4:0]]};
        endcase
    end

    wire signed [11:0] activation_ext = {{(12-DW){1'b0}}, activation};
    reg signed [11:0] product_small;
    always @(*) begin
        case ($signed(weight_data))
            -4'sd8: product_small = -(activation_ext <<< 3);
            -4'sd7: product_small = -(activation_ext <<< 3) + activation_ext;
            -4'sd6: product_small = -(activation_ext <<< 2) - (activation_ext <<< 1);
            -4'sd5: product_small = -(activation_ext <<< 2) - activation_ext;
            -4'sd4: product_small = -(activation_ext <<< 2);
            -4'sd3: product_small = -(activation_ext <<< 1) - activation_ext;
            -4'sd2: product_small = -(activation_ext <<< 1);
            -4'sd1: product_small = -activation_ext;
             4'sd0: product_small = 0;
             4'sd1: product_small = activation_ext;
             4'sd2: product_small = activation_ext <<< 1;
             4'sd3: product_small = (activation_ext <<< 1) + activation_ext;
             4'sd4: product_small = activation_ext <<< 2;
             4'sd5: product_small = (activation_ext <<< 2) + activation_ext;
             4'sd6: product_small = (activation_ext <<< 2) + (activation_ext <<< 1);
             4'sd7: product_small = (activation_ext <<< 3) - activation_ext;
            default: product_small = 0;
        endcase
    end
    wire signed [ACC_W-1:0] product =
        {{(ACC_W-12){product_small[11]}}, product_small};
    wire signed [ACC_W-1:0] sum = (col == 0) ? product : acc + product;

    wire accept = running && weight_valid &&
                  !(pending && (pending_row == last_row));
    wire row_end = accept && (col == last_col);
    wire pending_is_better = (pending_row == 0) || (acc > best_score);

    function [6:0] quantize_relu;
        input signed [ACC_W-1:0] value;
        input [4:0] shift;
        reg signed [ACC_W:0] nonnegative;
        reg signed [ACC_W:0] rounded;
        begin
            nonnegative = value[ACC_W-1] ? 0 : {value[ACC_W-1], value};
            rounded = (nonnegative + (1 <<< (shift - 1))) >>> shift;
            if (rounded > 127)
                quantize_relu = 7'd127;
            else if (rounded < 0)
                quantize_relu = 7'd0;
            else
                quantize_relu = rounded[6:0];
        end
    endfunction

    assign valid = valid_r;
    assign out = out_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running <= 1'b0;
            layer <= L1;
            row <= 0;
            col <= 0;
            pending_row <= 0;
            pending_layer <= 0;
            pending <= 1'b0;
            acc <= 0;
            best_score <= 0;
            best_index <= 0;
            valid_r <= 1'b0;
            out_r <= 0;
        end
        else begin
            valid_r <= 1'b0;

            if (!running) begin
                if (start) begin
                    running <= 1'b1;
                    layer <= L1;
                    row <= 0;
                    col <= 0;
                    pending <= 1'b0;
                    best_score <= 0;
                    best_index <= 0;
                end
            end
            else begin
                if (pending) begin
                    case (pending_layer)
                        L1: h1[pending_row] <= quantize_relu(acc, R1_SHIFT);
                        L2: h2[pending_row[4:0]] <= quantize_relu(acc, R2_SHIFT);
                        default: begin
                            if (pending_is_better) begin
                                best_score <= acc;
                                best_index <= pending_row[3:0];
                            end
                            if (pending_row == 9) begin
                                out_r <= pending_is_better ? pending_row[3:0] : best_index;
                                valid_r <= 1'b1;
                                running <= 1'b0;
                            end
                        end
                    endcase
                end

                if (accept) begin
                    acc <= sum;
                    if (row_end) begin
                        col <= 0;
                        pending <= 1'b1;
                        pending_row <= row;
                        pending_layer <= layer;
                        if (row != last_row)
                            row <= row + 1'b1;
                    end
                    else begin
                        col <= col + 1'b1;
                    end
                end

                if (pending && (pending_row == last_row)) begin
                    pending <= 1'b0;
                    if (pending_layer == L1) begin
                        layer <= L2;
                        row <= 0;
                        col <= 0;
                    end
                    else if (pending_layer == L2) begin
                        layer <= L3;
                        row <= 0;
                        col <= 0;
                    end
                end
            end
        end
    end
endmodule