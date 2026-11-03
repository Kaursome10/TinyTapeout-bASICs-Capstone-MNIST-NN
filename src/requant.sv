module requant #(
    parameter SHIFT,
    parameter ACC_W = 32,
    parameter QMAX = 8'd127
)(
    input  wire signed [ACC_W-1:0] acc,
    output wire [7:0]              h_out
);
    // Power-of-two scale factor: pure arithmetic right-shift, no multiplier.
    // acc is always >= 0 here (ReLU is applied upstream).
    wire signed [ACC_W-1:0] rounded = (acc + ({{(ACC_W-1){1'b0}}, 1'b1} << (SHIFT - 1))) >>> SHIFT;
    wire signed [ACC_W-1:0] clamped = (rounded > QMAX) ? QMAX : rounded;

    assign h_out = clamped[7:0];
endmodule

//