module mem #(
  parameter N = 8, // number of rows, N
  parameter M = 8, // number of cols, M
  parameter DW = 8,
  parameter INIT_FILE = ""
) (
  input [$clog2(N*M)-1 :0] raddr, 
  input [$clog2(N*M)-1 :0] waddr,
  input signed [DW-1 :0] wdata, 
  input write_enable, clk, 
  output signed [DW-1 :0] rdata
);

reg signed [DW-1 :0] memory [0: (N*M)-1];

always @(posedge clk) begin
  if(write_enable) begin
    memory[waddr] <= wdata;
  end
end

assign rdata = memory[raddr];

endmodule