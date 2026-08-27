module requant #(
    parameter SHIFT,
    parameter QMAX = 8'd127
)(
    input  wire signed [31:0] acc,
    output wire [7:0]         h_out
);
    wire signed [31:0] rounded = (acc + (32'sd1 << (SHIFT - 1))) >>> SHIFT;
    wire signed [31:0] clamped = (rounded > QMAX) ? QMAX : rounded;

    assign h_out = clamped[7:0];
endmodule