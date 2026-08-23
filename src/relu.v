module relu #(
    parameter W = 32
)(
    input  wire signed [W-1:0] in,
    output wire signed [W-1:0] out
);

    // Combinationally pass 0 if negative (sign bit is 1), else pass input
    assign out = in[W-1] ? {W{1'b0}} : in;

endmodule