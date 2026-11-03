module mac_unit #(
    parameter DW    = 8,   // activation (b) operand width
    parameter WW    = DW,  // weight (a) operand width
    parameter ACC_W = 32   // accumulator width
) (
  input signed [WW-1:0] a,
  input [DW-1:0] b,
  input valid_in,
  input clear_acc,
  input rst_n,
  input clk,
  output reg valid_out,
  output reg signed [ACC_W-1 :0] acc_out
);

reg signed [WW+DW-1:0] product;
reg signed [WW+DW-1:0] product_next;
reg prev_valid;
reg prev_clear;
wire signed [WW+DW-1:0] b_ext = {{WW{1'b0}}, b};

always @(*) begin
  // WW=4 weights are signed constants from -8 through 7. A case-based
  // constant multiply can map more compactly than a general multiplier.
  case ($signed(a))
    -4'sd8: product_next = -(b_ext <<< 3);
    -4'sd7: product_next = -(b_ext <<< 3) + b_ext;
    -4'sd6: product_next = -(b_ext <<< 2) - (b_ext <<< 1);
    -4'sd5: product_next = -(b_ext <<< 2) - b_ext;
    -4'sd4: product_next = -(b_ext <<< 2);
    -4'sd3: product_next = -(b_ext <<< 1) - b_ext;
    -4'sd2: product_next = -(b_ext <<< 1);
    -4'sd1: product_next = -b_ext;
     4'sd0: product_next = 0;
     4'sd1: product_next = b_ext;
     4'sd2: product_next = b_ext <<< 1;
     4'sd3: product_next = (b_ext <<< 1) + b_ext;
     4'sd4: product_next = b_ext <<< 2;
     4'sd5: product_next = (b_ext <<< 2) + b_ext;
     4'sd6: product_next = (b_ext <<< 2) + (b_ext <<< 1);
     4'sd7: product_next = (b_ext <<< 3) - b_ext;
    default: product_next = 0;
  endcase
end

always @(posedge clk or negedge rst_n) begin
  if (!rst_n) begin
    acc_out <= 0;
    valid_out <= 0;
    prev_valid <= 0;
    prev_clear <= 0;
    product <= 0;
  end
  else begin
    prev_valid <= valid_in;
    prev_clear <= clear_acc;
    product <= product_next;
    valid_out <= prev_valid;

    if(prev_valid && prev_clear) begin
      /* verilator lint_off WIDTHEXPAND */
      acc_out <= product;
      /* verilator lint_on WIDTHEXPAND */
    end
    else if (prev_valid && !prev_clear) begin
      /* verilator lint_off WIDTHEXPAND */
      acc_out <= acc_out + product;
      /* verilator lint_on WIDTHEXPAND */
    end
  end
end
endmodule