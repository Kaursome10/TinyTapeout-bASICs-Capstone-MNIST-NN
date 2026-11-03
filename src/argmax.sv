module argmax #(
    parameter N     = 10,
    parameter WIDTH = 32
)(
    input  wire signed [N*WIDTH-1:0] in_flat,
    output reg  [3:0] max_index
);
    integer i;
    reg signed [WIDTH-1:0] mv;
    reg signed [WIDTH-1:0] val;

    always @(*) begin
        mv = $signed(in_flat[WIDTH-1:0]); // index 0
        max_index = 4'd0;

        for (i = 1; i < N; i = i + 1) begin
            val = $signed(in_flat[i*WIDTH +: WIDTH]);
            if (val > mv) begin
                mv = val;
                max_index = i[3:0];
            end
        end
    end

endmodule
//
