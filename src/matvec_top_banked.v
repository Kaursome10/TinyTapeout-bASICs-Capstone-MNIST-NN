// Matrix-Vector multiplication: (N x M) x (M x 1) = (N x 1)
`timescale 1ns/1ps

module matvec_top_banked #(
    parameter N     = 8,   // number of rows in A / length of output vector
    parameter M     = 8,   // number of cols in A / length of input vector
    parameter DW    = 8,   // input operand width (signed INT8)
    parameter ACC_W = 32,  // accumulator / output width
    parameter INIT_FILE="", // init file for matrix; used when NBANKS==1

    // Set NBANKS=4 to split the weight ROM into 4 physically-separable
    // rom instances (see rtl/rom_banked.sv) instead of one monolithic
    // rom -- use this for layers whose GRT-0118 congestion isn't solved
    // by node-count alone. Leave at 1 (default) for unchanged behavior.
    parameter NBANKS = 1,
    parameter BANK0_FILE = "",
    parameter BANK1_FILE = "",
    parameter BANK2_FILE = "",
    parameter BANK3_FILE = ""
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,   // pulse 1 cycle: begin a new multiply
    output reg  done,    // pulses 1 cycle: full output valid

    // matrix/vector write
    input  wire ld_en,        // can data be loaded?
    input  wire ld_sel_mv,    // 0 = write matrix A, 1 = write vector x
    input  wire [$clog2(N*M)-1:0] ld_addr, // iC*M + kC for matrix, kC for vector; ignores ld_addr > M for vector write?
    input  wire signed [DW-1:0]   ld_data, // data to store in memory

    input  wire rd_en,
    input  wire [$clog2(N)-1:0]  rd_addr,
    output reg  signed [ACC_W-1:0] rd_data
);

    localparam IDLE = 0, COMPUTE = 1, DONE = 2;

    localparam MAC_LATENCY = 2;

    reg [1:0] state;

    // iC must represent 0..N (N+1 values) since it's used as a "past last
    // row" sentinel in `feeding`. kC only ever holds 0..M-1.
    reg [$clog2(N+1)-1:0] iC; // must count up to N, not N-1
    reg [$clog2(M)-1:0]   kC;

    reg [$clog2(N+1)-1:0] tC; // number of completed output rows

    wire [$clog2(N*M)-1:0] m_raddr;
    wire [$clog2(M)-1:0]   v_raddr;
    wire signed [DW-1:0]   m_rdata, v_rdata;
    assign m_raddr = iC * M + kC;   // A[i][k]
    assign v_raddr = kC;            // x[k]

    wire feeding    = (state == COMPUTE) && (iC < N);
    wire valid_in   = feeding;
    wire clear_acc  = feeding && (kC == 0);

    // Matrix M: N*M elements.
    // NBANKS==1 (default): single monolithic rom, same as before.
    // NBANKS==4: rom_banked splits the same N*M entries across 4
    // independent rom instances so placement/routing can spread them out
    // instead of clustering one giant read-decode mux tree in one spot.
    generate
        if (NBANKS == 4) begin : g_banked
            rom_banked #(.N(N), .M(M), .DW(DW),
                         .BANK0_FILE(BANK0_FILE), .BANK1_FILE(BANK1_FILE),
                         .BANK2_FILE(BANK2_FILE), .BANK3_FILE(BANK3_FILE)) mem_M (
                .raddr (m_raddr),
                .rdata (m_rdata)
            );
        end else begin : g_single
            rom #(.N(N), .M(M), .DW(DW), .INIT_FILE(INIT_FILE)) mem_M (
                .raddr (m_raddr),
                .rdata (m_rdata)
            );
        end
    endgenerate

    // Vector x: M elements
    mem #(.N(1'b1), .M(M), .DW(DW)) mem_V (
        .clk          (clk),
        .write_enable (ld_en && ld_sel_mv),
        .waddr        (ld_addr[$clog2(M)-1:0]),
        .wdata        (ld_data),
        .raddr        (v_raddr),
        .rdata        (v_rdata)
    );

    // MAC (Module 1)
    wire                    mac_valid_out;
    wire signed [ACC_W-1:0] mac_acc_out;

    mac_unit #(.DW(DW), .ACC_W(ACC_W)) mac0 (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .clear_acc (clear_acc),
        .a         (m_rdata),
        .b         (v_rdata),
        .valid_out (mac_valid_out),
        .acc_out   (mac_acc_out)
    );

    reg [$clog2(N)-1:0] addr_sr [0:MAC_LATENCY-1];
    reg                 last_sr [0:MAC_LATENCY-1];
    integer s;

    always @(posedge clk) begin
        addr_sr[0] <= iC[$clog2(N)-1:0];
        last_sr[0] <= (kC == M-1);
        for (s = 1; s < MAC_LATENCY; s = s + 1) begin
            addr_sr[s] <= addr_sr[s-1];
            last_sr[s] <= last_sr[s-1];
        end
    end

    wire [$clog2(N)-1:0] mac_addr_out = addr_sr[MAC_LATENCY-1];
    wire                 mac_last_out = last_sr[MAC_LATENCY-1];

    wire signed [ACC_W-1:0] c_rdata;

    // Output vector C: N elements
    mem #(.N(N), .M(1), .DW(ACC_W)) mem_C (
        .clk          (clk),
        .write_enable (mac_valid_out && mac_last_out),
        .waddr        (mac_addr_out),
        .wdata        (mac_acc_out),
        .raddr        (rd_addr),
        .rdata        (c_rdata)
    );


    // delay for 1 cycle
    always @(*) begin
        rd_data = c_rdata;
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            iC    <= 0;
            kC    <= 0;
            tC    <= 0;
            done  <= 1'b0;
        end
        else begin
            case (state)
                IDLE : begin
                    done <= 0;
                    if (start) begin
                        iC    <= 0;
                        kC    <= 0;
                        tC    <= 0;
                        state <= COMPUTE;
                    end
                end

                COMPUTE : begin
                    done <= 1'b0;

                    if (iC < N) begin
                        if (kC == M-1) begin
                            kC <= 0;
                            iC <= iC + 1;
                        end
                        else begin
                            kC <= kC + 1;
                        end
                    end

                    if (mac_valid_out && mac_last_out) begin
                        tC <= tC + 1;
                    end
                    if (mac_valid_out && mac_last_out && (tC == N - 1)) begin
                        state <= DONE;
                    end
                end

                DONE : begin
                    done  <= 1'b1; // pulses exactly one cycle
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule