// 4-bank version of rom.sv.
//
// Why this exists: yosys's MEMORY_MAP pass turns a $readmemh reg array
// into flip-flops + a full binary read-decode mux tree sized to the next
// power of two above N*M. For a single N*M=7840-entry ROM that's a
// monolithic ~8191-mux block that placement clusters in one spot and the
// router can't get through (GRT-0118). Splitting the same N*M entries
// across 4 independent rom instances gives 4 much smaller mux trees
// (~2 fewer levels of tree depth each) that placement is free to spread
// across the die instead of clustering into a single congestion hotspot.
//
// Row-major layout is preserved (raddr = row*M + col, same as rom.sv),
// so nothing else about matvec_top's addressing needs to change -- this
// is meant to be a drop-in replacement for `rom` inside matvec_top.

module rom_banked #(
    parameter N  = 8,   // number of rows (must be evenly divisible by 4)
    parameter M  = 8,   // number of cols
    parameter DW = 8,
    parameter BANK0_FILE = "",
    parameter BANK1_FILE = "",
    parameter BANK2_FILE = "",
    parameter BANK3_FILE = ""
) (
    input  wire [$clog2(N*M)-1:0] raddr,
    output wire signed [DW-1:0]   rdata
);

    localparam ROWS_PER_BANK  = N / 4;                 // rows per bank
    localparam BANK_SPAN      = ROWS_PER_BANK * M;      // addresses per bank
    localparam BANK_ADDR_W    = $clog2(BANK_SPAN);

    // Which bank this address falls in, and the address local to that bank.
    // BANK_SPAN is a compile-time constant, so this division/mod is cheap
    // (shift-add), not a real divider -- and it's tiny next to the mux
    // tree it's replacing.
    wire [1:0]              bank_sel  = raddr / BANK_SPAN;
    wire [BANK_ADDR_W-1:0]  bank_addr = raddr % BANK_SPAN;

    wire signed [DW-1:0] rdata0, rdata1, rdata2, rdata3;

    rom #(.N(ROWS_PER_BANK), .M(M), .DW(DW), .INIT_FILE(BANK0_FILE)) bank0 (
        .raddr (bank_addr),
        .rdata (rdata0)
    );
    rom #(.N(ROWS_PER_BANK), .M(M), .DW(DW), .INIT_FILE(BANK1_FILE)) bank1 (
        .raddr (bank_addr),
        .rdata (rdata1)
    );
    rom #(.N(ROWS_PER_BANK), .M(M), .DW(DW), .INIT_FILE(BANK2_FILE)) bank2 (
        .raddr (bank_addr),
        .rdata (rdata2)
    );
    rom #(.N(ROWS_PER_BANK), .M(M), .DW(DW), .INIT_FILE(BANK3_FILE)) bank3 (
        .raddr (bank_addr),
        .rdata (rdata3)
    );

    // Cheap 4:1 output select -- 4 inputs vs. the ~8191-entry tree this
    // replaces internally per bank.
    assign rdata = (bank_sel == 2'd0) ? rdata0 :
                   (bank_sel == 2'd1) ? rdata1 :
                   (bank_sel == 2'd2) ? rdata2 :
                                        rdata3;

endmodule