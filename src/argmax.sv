module argmax (
    // Packed bus: in_flat[32*i +: 32] corresponds to index i (i=0..9)
    input  wire signed [319:0] in_flat,
    output reg  [3:0] max_index
);

    integer i;
    reg signed [31:0] mv;
    reg signed [31:0] val;

    always @(*) begin
        mv = $signed(in_flat[31:0]); // index 0
        max_index = 4'd0;

        for (i = 1; i < 10; i = i + 1) begin
            val = $signed(in_flat[i*32 +: 32]);
            if (val > mv) begin
                mv = val;
                max_index = i[3:0];
            end
        end
    end

endmodule