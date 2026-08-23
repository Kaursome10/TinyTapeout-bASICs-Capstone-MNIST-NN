module rom #(
  parameter INIT_FILE,
  parameter N = 8, // number of rows, N
  parameter M = 8, // number of cols, M
  parameter DW = 8
) (
  input [$clog2(N*M)-1 :0] raddr, 
  output signed [DW-1 :0] rdata
);

reg signed [DW-1 :0] memory [0: (N*M)-1];

initial begin
  if(INIT_FILE != "") $readmemh(INIT_FILE, memory);
end

assign rdata = memory[raddr];

endmodule