module requant #(
    parameter MULT, 
    parameter SHIFT, 
    parameter QMAX = 8'd127 
)(
    input  wire signed [31:0] acc,     
    output wire [7:0]         h_out    
);
    wire [47:0] product;
    wire [31:0] rounded;
    wire [31:0] clamped;

    assign product = acc * $signed(MULT);                           

    // Proper round-to-nearest (using 48-bit literal to prevent any 32-bit overflow)
    assign rounded = (product + (48'd1 << (SHIFT - 1))) >> SHIFT;

    assign clamped = (rounded > QMAX) ? QMAX : rounded;

    assign h_out = clamped[7:0];
endmodule