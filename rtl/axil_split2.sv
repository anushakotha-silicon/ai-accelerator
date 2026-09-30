// AXI4-Lite 1-to-2 router: address bit SEL_BIT picks the target.
// One transaction in flight per direction; AW and W may arrive in any order.
module axil_split2 #(
  parameter int SEL_BIT = 22
) (
  input  logic        clk,
  input  logic        rst_n,
  // from the host
  input  logic [31:0] s_awaddr,  input  logic s_awvalid, output logic s_awready,
  input  logic [31:0] s_wdata,   input  logic s_wvalid,  output logic s_wready,
  output logic [1:0]  s_bresp,   output logic s_bvalid,  input  logic s_bready,
  input  logic [31:0] s_araddr,  input  logic s_arvalid, output logic s_arready,
  output logic [31:0] s_rdata,   output logic [1:0] s_rresp, output logic s_rvalid, input logic s_rready,
  // to target 0 (bit clear) and target 1 (bit set)
  output logic [31:0] m_awaddr,  output logic [1:0] m_awvalid, input logic [1:0] m_awready,
  output logic [31:0] m_wdata,   output logic [1:0] m_wvalid,  input logic [1:0] m_wready,
  input  logic [3:0]  m_bresp,   input  logic [1:0] m_bvalid,  output logic [1:0] m_bready,
  output logic [31:0] m_araddr,  output logic [1:0] m_arvalid, input logic [1:0] m_arready,
  input  logic [63:0] m_rdata,   input  logic [3:0] m_rresp,   input logic [1:0] m_rvalid,
  output logic [1:0]  m_rready
);
  // ---------------------------------------------------------------- write
  typedef enum logic [1:0] {W_IDLE, W_FWD, W_RESP, W_DONE} wst_t;
  wst_t        wst;
  logic        aw_have, w_have, aw_sent, w_sent, wsel;
  logic [31:0] aw_q, w_q;
  logic [1:0]  b_q;

  assign s_awready = (wst == W_IDLE) && !aw_have;
  assign s_wready  = (wst == W_IDLE) && !w_have;
  assign m_awaddr  = aw_q;
  assign m_wdata   = w_q;
  assign m_awvalid = (wst == W_FWD && !aw_sent) ? (2'b01 << wsel) : 2'b00;
  assign m_wvalid  = (wst == W_FWD && !w_sent)  ? (2'b01 << wsel) : 2'b00;
  assign m_bready  = (wst == W_RESP)            ? (2'b01 << wsel) : 2'b00;
  assign s_bvalid  = (wst == W_DONE);
  assign s_bresp   = b_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wst <= W_IDLE; aw_have <= 1'b0; w_have <= 1'b0; aw_sent <= 1'b0; w_sent <= 1'b0;
      wsel <= 1'b0; aw_q <= '0; w_q <= '0; b_q <= '0;
    end else case (wst)
      W_IDLE: begin
        if (s_awvalid && s_awready) begin aw_have <= 1'b1; aw_q <= s_awaddr; wsel <= s_awaddr[SEL_BIT]; end
        if (s_wvalid && s_wready)   begin w_have  <= 1'b1; w_q  <= s_wdata; end
        if ((aw_have || (s_awvalid && s_awready)) && (w_have || (s_wvalid && s_wready))) begin
          wst <= W_FWD; aw_sent <= 1'b0; w_sent <= 1'b0;
        end
      end
      W_FWD: begin
        if (m_awvalid[wsel] && m_awready[wsel]) aw_sent <= 1'b1;
        if (m_wvalid[wsel]  && m_wready[wsel])  w_sent  <= 1'b1;
        if ((aw_sent || (m_awvalid[wsel] && m_awready[wsel])) &&
            (w_sent  || (m_wvalid[wsel]  && m_wready[wsel]))) wst <= W_RESP;
      end
      W_RESP: if (m_bvalid[wsel]) begin b_q <= m_bresp[wsel*2 +: 2]; wst <= W_DONE; end
      W_DONE: if (s_bready) begin wst <= W_IDLE; aw_have <= 1'b0; w_have <= 1'b0; end
    endcase
  end

  // ----------------------------------------------------------------- read
  typedef enum logic [1:0] {R_IDLE, R_FWD, R_WAIT, R_DONE} rst_t;
  rst_t        rst;
  logic        rsel;
  logic [31:0] ar_q, r_q;
  logic [1:0]  rr_q;

  assign s_arready = (rst == R_IDLE);
  assign m_araddr  = ar_q;
  assign m_arvalid = (rst == R_FWD)  ? (2'b01 << rsel) : 2'b00;
  assign m_rready  = (rst == R_WAIT) ? (2'b01 << rsel) : 2'b00;
  assign s_rvalid  = (rst == R_DONE);
  assign s_rdata   = r_q;
  assign s_rresp   = rr_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rst <= R_IDLE; rsel <= 1'b0; ar_q <= '0; r_q <= '0; rr_q <= '0;
    end else case (rst)
      R_IDLE: if (s_arvalid) begin ar_q <= s_araddr; rsel <= s_araddr[SEL_BIT]; rst <= R_FWD; end
      R_FWD:  if (m_arready[rsel]) rst <= R_WAIT;
      R_WAIT: if (m_rvalid[rsel]) begin r_q <= m_rdata[rsel*32 +: 32]; rr_q <= m_rresp[rsel*2 +: 2]; rst <= R_DONE; end
      R_DONE: if (s_rready) rst <= R_IDLE;
    endcase
  end
endmodule
