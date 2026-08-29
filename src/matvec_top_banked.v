// Matrix-Vector multiplication: (N x M) x (M x 1) = (N x 1)
`timescale 1ns/1ps

module matvec_top_banked #(
    parameter N     = 8,   // number of rows in A / length of output vector
    parameter M     = 8,   // number of cols in A / length of input vector
    parameter DW    = 8,   // input operand wire width (pins/buses stay this wide)
    parameter WW    = DW,  // meaningful weight bits actually used (<=DW)
    parameter ACC_W = 32   // accumulator / output width
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,   // pulse 1 cycle: begin a new multiply
    output reg  done,    // pulses 1 cycle: full output valid

    // vector write (unchanged from before)
    input  wire ld_en,
    input  wire [$clog2(M)-1:0]   ld_addr,
    input  wire signed [DW-1:0]   ld_data,

    // streamed weight input: one matrix element per pulse, row-major order
    input  wire                   weight_valid,
    input  wire signed [WW-1:0]   weight_data,

    input  wire rd_en,
    input  wire [$clog2(N)-1:0]  rd_addr,
    output reg  signed [ACC_W-1:0] rd_data
);

    localparam IDLE = 0, COMPUTE = 1, DONE = 2;
    localparam MAC_LATENCY = 2;

    reg [1:0] state;

    reg [$clog2(N+1)-1:0] iC; // must count up to N, not N-1
    reg [$clog2(M)-1:0]   kC;
    reg [$clog2(N+1)-1:0] tC; // number of completed output rows

    wire [$clog2(M)-1:0]   v_raddr;
    wire signed [DW-1:0]   v_rdata;
    assign v_raddr = kC;            // x[k]

    // Only advance the schedule -- and only let the MAC consume this
    // cycle's data -- when a weight byte is actually present.
    wire feeding    = (state == COMPUTE) && (iC < N) && weight_valid;
    wire valid_in   = feeding;
    wire clear_acc  = feeding && (kC == 0);

    // Matrix element comes straight from the top-level port now.
    wire signed [WW-1:0] m_rdata = weight_data;

    // Vector x: M elements (unchanged)
    mem #(.N(1'b1), .M(M), .DW(DW)) mem_V (
        .clk          (clk),
        .write_enable (ld_en),
        .waddr        (ld_addr),
        .wdata        (ld_data),
        .raddr        (v_raddr),
        .rdata        (v_rdata)
    );

    // MAC (unchanged)
    wire                    mac_valid_out;
    wire signed [ACC_W-1:0] mac_acc_out;

    mac_unit #(.DW(DW), .WW(WW), .ACC_W(ACC_W)) mac0 (
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

    // Output vector C: N elements (unchanged)
    mem #(.N(N), .M(1), .DW(ACC_W)) mem_C (
        .clk          (clk),
        .write_enable (mac_valid_out && mac_last_out),
        .waddr        (mac_addr_out),
        .wdata        (mac_acc_out),
        .raddr        (rd_addr),
        .rdata        (c_rdata)
    );

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

                    // Only step forward on cycles where a weight actually arrived.
                    if (feeding) begin
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
                    done  <= 1'b1;
                    state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule