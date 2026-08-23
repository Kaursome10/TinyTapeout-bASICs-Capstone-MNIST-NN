module matmul_top #(
    parameter N     = 8, // matrix dimension, N x N
    parameter DW    = 8, // input operand width (signed INT8)
    parameter ACC_W = 32 // accumulator / output width
) (
    input  wire clk,
    input  wire rst_n,

    input  wire start,   // pulse 1 cycle: begin a new multiply
    output reg done,    // pulses 1 cycle: full output valid

    input  wire ld_en, //can data be loaded?
    input  wire ld_sel_ab, // 0 = write A, 1 = write B
    input  wire [$clog2(N*N)-1:0] ld_addr, //address in memory to load data
    input  wire signed [DW-1:0] ld_data, //data to store in memory

    input  wire rd_en,
    input  wire [$clog2(N*N)-1:0] rd_addr,
    output reg  signed [ACC_W-1:0] rd_data
);

    localparam IDLE = 0, COMPUTE = 1, DONE = 2;

    localparam MAC_LATENCY = 2;

    reg [1:0] state;

    // iC must represent 0..N (N+1 values) since it's used as a "past last
    // row" sentinel in `feeding`. jC/kC only ever hold 0..N-1.
    reg [$clog2(N+1)-1:0] iC; //must count up to N, not N-1
    reg [$clog2(N)-1:0]   jC, kC;

    reg [$clog2(N*N+1)-1 :0] tC;

    wire [$clog2(N*N)-1:0] a_raddr, b_raddr;
    wire signed [DW-1:0]   a_rdata, b_rdata;
    assign a_raddr = iC * N + kC;   // A[i][k]
    assign b_raddr = kC * N + jC;   // B[k][j]

    wire feeding    = (state == COMPUTE) && (iC < N);
    wire valid_in   = feeding;
    wire clear_acc  = feeding && (kC == 0);

    mem #(.N(N), .DW(DW)) mem_A (
        .clk          (clk),
        .write_enable (ld_en && !ld_sel_ab),
        .waddr        (ld_addr),
        .wdata        (ld_data),
        .raddr        (a_raddr),
        .rdata        (a_rdata)
    );

    mem #(.N(N), .DW(DW)) mem_B (
        .clk          (clk),
        .write_enable (ld_en && ld_sel_ab),
        .waddr        (ld_addr),
        .wdata        (ld_data),
        .raddr        (b_raddr),
        .rdata        (b_rdata)
    );

    // MAC (Module 1)
    wire                    mac_valid_out;
    wire signed [ACC_W-1:0] mac_acc_out;

    mac_unit #(.DW(DW), .ACC_W(ACC_W)) mac0 (
        .clk       (clk),
        .rst_n     (rst_n),
        .valid_in  (valid_in),
        .clear_acc (clear_acc),
        .a         (a_rdata),
        .b         (b_rdata),
        .valid_out (mac_valid_out),
        .acc_out   (mac_acc_out)
    );

    reg [$clog2(N*N)-1:0] addr_sr [0:MAC_LATENCY-1];
    reg                   last_sr [0:MAC_LATENCY-1];
    integer s;

    always @(posedge clk) begin
        addr_sr[0] <= iC * N + jC;
        last_sr[0] <= (kC == N-1);
        for (s = 1; s < MAC_LATENCY; s = s + 1) begin
            addr_sr[s] <= addr_sr[s-1];
            last_sr[s] <= last_sr[s-1];
        end
    end

    wire [$clog2(N*N)-1:0] mac_addr_out = addr_sr[MAC_LATENCY-1];
    wire                   mac_last_out = last_sr[MAC_LATENCY-1];

    wire signed [ACC_W-1:0] c_rdata;

    mem #(.N(N), .DW(ACC_W)) mem_C (
        .clk          (clk),
        .write_enable (mac_valid_out),
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
            jC    <= 0;
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
                        jC    <= 0;
                        kC    <= 0;
                        tC    <= 0;
                        state <= COMPUTE;
                    end
                end

                COMPUTE : begin
                    done <= 1'b0;

                    if (iC < N) begin
                        if (kC == N-1) begin
                            kC <= 0;
                            if (jC == N-1) begin
                                jC <= 0;
                                iC <= iC + 1;
                            end
                            else begin
                                jC <= jC + 1;
                            end
                        end
                        else begin
                            kC <= kC + 1;
                        end
                    end

                    if (mac_valid_out && mac_last_out) begin
                        tC <= tC + 1;
                    end
                    if (mac_valid_out && mac_last_out && (tC == N*N - 1)) begin
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