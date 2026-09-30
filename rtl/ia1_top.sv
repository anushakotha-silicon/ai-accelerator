// IA-1 FPGA top: one AXI4-Lite port (F2 OCL / BAR0) shared by the tile core
// and the agent KV manager.
//   0x000000 - 0x3FFFFF  tile_core   (address bit 22 clear; map in rtl/tile_core.sv)
//   0x400000 - 0x4FFFFF  kv_axil     (address bit 22 set;   map in rtl/kv_axil.sv)
module ia1_top #(
  parameter int N = 32, LANES = 2, MAX_BLOCKS = 16, MAX_M = 64,
  parameter int KV_S = 8, KV_P = 16, KV_PAGE_TOK = 16, KV_H = 32, KV_D = 128
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] s_awaddr,  input  logic s_awvalid, output logic s_awready,
  input  logic [31:0] s_wdata,   input  logic s_wvalid,  output logic s_wready,
  output logic [1:0]  s_bresp,   output logic s_bvalid,  input  logic s_bready,
  input  logic [31:0] s_araddr,  input  logic s_arvalid, output logic s_arready,
  output logic [31:0] s_rdata,   output logic [1:0] s_rresp, output logic s_rvalid, input logic s_rready
);
  logic [31:0] awaddr, wdata, araddr;
  logic [1:0]  awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
  logic [3:0]  bresp, rresp;
  logic [63:0] rdata;

  axil_split2 #(.SEL_BIT(22)) u_split (
    .clk(clk), .rst_n(rst_n),
    .s_awaddr(s_awaddr), .s_awvalid(s_awvalid), .s_awready(s_awready),
    .s_wdata(s_wdata), .s_wvalid(s_wvalid), .s_wready(s_wready),
    .s_bresp(s_bresp), .s_bvalid(s_bvalid), .s_bready(s_bready),
    .s_araddr(s_araddr), .s_arvalid(s_arvalid), .s_arready(s_arready),
    .s_rdata(s_rdata), .s_rresp(s_rresp), .s_rvalid(s_rvalid), .s_rready(s_rready),
    .m_awaddr(awaddr), .m_awvalid(awvalid), .m_awready(awready),
    .m_wdata(wdata), .m_wvalid(wvalid), .m_wready(wready),
    .m_bresp(bresp), .m_bvalid(bvalid), .m_bready(bready),
    .m_araddr(araddr), .m_arvalid(arvalid), .m_arready(arready),
    .m_rdata(rdata), .m_rresp(rresp), .m_rvalid(rvalid), .m_rready(rready)
  );

  tile_core #(.N(N), .LANES(LANES), .MAX_BLOCKS(MAX_BLOCKS), .MAX_M(MAX_M)) u_tile (
    .clk(clk), .rst_n(rst_n),
    .s_awaddr(awaddr), .s_awvalid(awvalid[0]), .s_awready(awready[0]),
    .s_wdata(wdata), .s_wvalid(wvalid[0]), .s_wready(wready[0]),
    .s_bresp(bresp[1:0]), .s_bvalid(bvalid[0]), .s_bready(bready[0]),
    .s_araddr(araddr), .s_arvalid(arvalid[0]), .s_arready(arready[0]),
    .s_rdata(rdata[31:0]), .s_rresp(rresp[1:0]), .s_rvalid(rvalid[0]), .s_rready(rready[0])
  );

  kv_axil #(.S(KV_S), .P(KV_P), .PAGE_TOK(KV_PAGE_TOK), .H(KV_H), .D(KV_D)) u_kv (
    .clk(clk), .rst_n(rst_n),
    .s_awaddr(awaddr), .s_awvalid(awvalid[1]), .s_awready(awready[1]),
    .s_wdata(wdata), .s_wvalid(wvalid[1]), .s_wready(wready[1]),
    .s_bresp(bresp[3:2]), .s_bvalid(bvalid[1]), .s_bready(bready[1]),
    .s_araddr(araddr), .s_arvalid(arvalid[1]), .s_arready(arready[1]),
    .s_rdata(rdata[63:32]), .s_rresp(rresp[3:2]), .s_rvalid(rvalid[1]), .s_rready(rready[1])
  );
endmodule
