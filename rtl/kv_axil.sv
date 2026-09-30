// AXI4-Lite front end for kv_manager: host registers -> KV commands.
//
// Host flow: write KV_TOK / KV_WDATA_LO / KV_WDATA_HI as needed, then KV_CMD;
// poll KV_STATUS.done; read KV_RVAL_* and KV_STATUS.code. KV_CYCLES holds how
// many clock cycles the last command took (park/restore latency in hardware).
//
// Register offsets (low 8 address bits; the router maps this block at 0x400000):
//   0x00 KV_ID        RO 0x1A1C_4B56
//   0x04 KV_CMD       WO [2:0] op, [7:4] session, [11:8] source, [20:16] n  (starts the command)
//   0x08 KV_TOK       RW token index for WRITE/READ
//   0x0C KV_WDATA_LO  RW
//   0x10 KV_WDATA_HI  RW
//   0x14 KV_STATUS    RO bit0 busy, bit1 done, [5:4] status code (0 OK, 1 NOSPACE, 2 RANGE, 3 STATE)
//   0x18 KV_RVAL_LO   RO response value
//   0x1C KV_RVAL_HI   RO
//   0x20 KV_PARKED    RO total tokens copied HBM -> DDR
//   0x24 KV_RESTORED  RO total tokens copied DDR -> HBM
//   0x28 KV_FREE      RO [15:0] free HBM pages, [31:16] free DDR pages
//   0x2C KV_CYCLES    RO cycles taken by the last command
//   0x30 KV_PARAMS    RO S | P<<8 | PAGE_TOK<<16
//   0x34 KV_TIERS     RO H | D<<16 (HBM and DDR pages)
// A KV_CMD write while busy is ignored.
module kv_axil #(
  parameter int S = 8, P = 16, PAGE_TOK = 16, H = 32, D = 128
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] s_awaddr,  input  logic s_awvalid, output logic s_awready,
  input  logic [31:0] s_wdata,   input  logic s_wvalid,  output logic s_wready,
  output logic [1:0]  s_bresp,   output logic s_bvalid,  input  logic s_bready,
  input  logic [31:0] s_araddr,  input  logic s_arvalid, output logic s_arready,
  output logic [31:0] s_rdata,   output logic [1:0] s_rresp, output logic s_rvalid, input logic s_rready
);
  localparam int SW = $clog2(S), PW = $clog2(P), TW = $clog2(P * PAGE_TOK);

  // ------------------------------------------------------------ kv_manager
  logic          cmd_valid, cmd_ready, rsp_valid, rsp_ready;
  logic [2:0]    cmd_op;
  logic [SW-1:0] cmd_sess, cmd_src;
  logic [PW:0]   cmd_n;
  logic [TW-1:0] cmd_tok;
  logic [63:0]   cmd_wdata, rsp_value;
  logic [1:0]    rsp_status;
  logic [31:0]   parked, restored;
  logic [$clog2(H):0] hbm_free;
  logic [$clog2(D):0] ddr_free;

  kv_manager #(.S(S), .P(P), .PAGE_TOK(PAGE_TOK), .H(H), .D(D), .DATA_W(64)) u_kv (
    .clk(clk), .rst_n(rst_n),
    .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_op(cmd_op), .cmd_sess(cmd_sess),
    .cmd_src(cmd_src), .cmd_n(cmd_n), .cmd_tok(cmd_tok), .cmd_wdata(cmd_wdata),
    .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_status(rsp_status), .rsp_value(rsp_value),
    .stat_tok_parked(parked), .stat_tok_restored(restored),
    .stat_hbm_free(hbm_free), .stat_ddr_free(ddr_free)
  );

  // ------------------------------------------------------------ registers
  logic [31:0] tok_q, wlo_q, whi_q, cycles, last_cycles;
  logic [63:0] rval_q;
  logic [1:0]  code_q;
  logic        busy, done;

  // ----------------------------------------------------- AXI-Lite write
  logic        aw_hold, w_hold;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [31:0] aw_q, w_q;                 // only the low 8 address bits are decoded
  /* verilator lint_on UNUSEDSIGNAL */
  wire         wr = aw_hold && w_hold && !s_bvalid;
  wire         launch = wr && aw_q[7:0] == 8'h04 && !busy;

  assign s_awready = !aw_hold;
  assign s_wready  = !w_hold;
  assign s_bresp   = 2'b00;
  assign rsp_ready = 1'b1;                // responses are captured the cycle they appear

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      aw_hold <= 1'b0; w_hold <= 1'b0; s_bvalid <= 1'b0; aw_q <= '0; w_q <= '0;
      tok_q <= '0; wlo_q <= '0; whi_q <= '0;
      cmd_valid <= 1'b0; cmd_op <= '0; cmd_sess <= '0; cmd_src <= '0; cmd_n <= '0;
      cmd_tok <= '0; cmd_wdata <= '0;
      busy <= 1'b0; done <= 1'b0; cycles <= '0; last_cycles <= '0; rval_q <= '0; code_q <= '0;
    end else begin
      if (s_awvalid && s_awready) begin aw_hold <= 1'b1; aw_q <= s_awaddr; end
      if (s_wvalid && s_wready)   begin w_hold  <= 1'b1; w_q  <= s_wdata;  end
      if (wr) begin
        aw_hold <= 1'b0; w_hold <= 1'b0; s_bvalid <= 1'b1;
        case (aw_q[7:0])
          8'h08: tok_q <= w_q;
          8'h0C: wlo_q <= w_q;
          8'h10: whi_q <= w_q;
          default: ;
        endcase
      end
      if (s_bvalid && s_bready) s_bvalid <= 1'b0;

      if (launch) begin
        cmd_valid <= 1'b1;
        cmd_op    <= w_q[2:0];
        cmd_sess  <= w_q[4 +: SW];
        cmd_src   <= w_q[8 +: SW];
        cmd_n     <= w_q[16 +: PW + 1];
        cmd_tok   <= tok_q[TW-1:0];
        cmd_wdata <= {whi_q, wlo_q};
        busy <= 1'b1; done <= 1'b0; cycles <= '0;
      end else if (busy) begin
        cycles <= cycles + 1;
        if (cmd_valid && cmd_ready) cmd_valid <= 1'b0;
        if (rsp_valid) begin
          busy <= 1'b0; done <= 1'b1;
          rval_q <= rsp_value; code_q <= rsp_status; last_cycles <= cycles + 1;
        end
      end
    end
  end

  // ------------------------------------------------------ AXI-Lite read
  logic        rd_pend;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [31:0] ar_q;
  /* verilator lint_on UNUSEDSIGNAL */
  assign s_arready = !rd_pend && !s_rvalid;
  assign s_rresp   = 2'b00;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_pend <= 1'b0; s_rvalid <= 1'b0; s_rdata <= '0; ar_q <= '0;
    end else begin
      if (s_arvalid && s_arready) begin rd_pend <= 1'b1; ar_q <= s_araddr; end
      if (rd_pend) begin
        rd_pend <= 1'b0; s_rvalid <= 1'b1;
        case (ar_q[7:0])
          8'h00:   s_rdata <= 32'h1A1C_4B56;
          8'h08:   s_rdata <= tok_q;
          8'h0C:   s_rdata <= wlo_q;
          8'h10:   s_rdata <= whi_q;
          8'h14:   s_rdata <= {26'd0, code_q, 2'b00, done, busy};
          8'h18:   s_rdata <= rval_q[31:0];
          8'h1C:   s_rdata <= rval_q[63:32];
          8'h20:   s_rdata <= parked;
          8'h24:   s_rdata <= restored;
          8'h28:   s_rdata <= {16'(ddr_free), 16'(hbm_free)};
          8'h2C:   s_rdata <= last_cycles;
          8'h30:   s_rdata <= 32'(S) | (32'(P) << 8) | (32'(PAGE_TOK) << 16);
          8'h34:   s_rdata <= 32'(H) | (32'(D) << 16);
          default: s_rdata <= 32'hDEAD_BEEF;
        endcase
      end
      if (s_rvalid && s_rready) s_rvalid <= 1'b0;
    end
  end
endmodule
