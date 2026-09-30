// End-to-end test of the FPGA top (ia1_top) through its single AXI4-Lite port,
// exactly as the F2 host drives it: routing, a tile matmul, and a multi-agent
// KV scenario through the KV registers with hardware cycle counts.
`timescale 1ns/1ps
`ifndef VEC_DIR
  `define VEC_DIR "build"
`endif
module tb_ia1_top;
  `include "params.svh"                        // N, LANES, M, NB for the tile test
  localparam int AW = 8, MAX_BLOCKS = 16, MAX_M = 64, WPR = N * AW / 32;
  localparam int KS = 8, KP = 16, PT = 16, KH = 32, KD = 128;
  localparam logic [31:0] KV = 32'h40_0000;
  localparam int ALLOC = 1, WRITE = 2, READ = 3, PARK = 4, RESTORE = 5, FREE = 6, SHARE = 7;
  localparam int AGENTS = 4, PREFIX = 2;

  logic clk = 1'b0, rst_n = 1'b0;
  always #2 clk = ~clk;

  logic [31:0] awaddr, wdata, araddr, rdata;
  logic awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
  logic [1:0] bresp, rresp;

  ia1_top #(.N(N), .LANES(LANES), .MAX_BLOCKS(MAX_BLOCKS), .MAX_M(MAX_M),
            .KV_S(KS), .KV_P(KP), .KV_PAGE_TOK(PT), .KV_H(KH), .KV_D(KD)) dut (
    .clk(clk), .rst_n(rst_n),
    .s_awaddr(awaddr), .s_awvalid(awvalid), .s_awready(awready),
    .s_wdata(wdata), .s_wvalid(wvalid), .s_wready(wready),
    .s_bresp(bresp), .s_bvalid(bvalid), .s_bready(bready),
    .s_araddr(araddr), .s_arvalid(arvalid), .s_arready(arready),
    .s_rdata(rdata), .s_rresp(rresp), .s_rvalid(rvalid), .s_rready(rready)
  );

  logic [7:0]  wmem [0:NB*N*N-1];
  logic [7:0]  xmem [0:NB*M*N-1];
  logic [31:0] ymem [0:NB*M*N-1];
  integer errors, checks;

  // ------------------------------------------------------------ AXI-Lite BFM
  task automatic axi_write(input logic [31:0] addr, input logic [31:0] data);
    @(negedge clk);
    awaddr = addr; awvalid = 1; wdata = data; wvalid = 1; bready = 1;
    fork
      begin do @(posedge clk); while (!awready); @(negedge clk); awvalid = 0; end
      begin do @(posedge clk); while (!wready);  @(negedge clk); wvalid  = 0; end
    join
    do @(posedge clk); while (!bvalid);
    @(negedge clk); bready = 0;
  endtask

  task automatic axi_read(input logic [31:0] addr, output logic [31:0] data);
    @(negedge clk);
    araddr = addr; arvalid = 1; rready = 1;
    do @(posedge clk); while (!arready);
    @(negedge clk); arvalid = 0;
    do @(posedge clk); while (!rvalid);
    data = rdata;
    @(negedge clk); rready = 0;
  endtask

  task automatic check(input string what, input logic [63:0] got, input logic [63:0] want);
    checks++;
    if (got !== want) begin
      errors++;
      if (errors <= 8) $display("ERROR %s: got %0h expected %0h", what, got, want);
    end
  endtask

  // ------------------------------------------------------------- KV helpers
  integer last_cycles;
  task automatic kv_cmd(input int op, input int s, input int src, input int n, input int tok,
                        input logic [63:0] wd, output int code, output logic [63:0] val);
    logic [31:0] st, lo, hi, cy;
    axi_write(KV + 32'h08, tok);
    axi_write(KV + 32'h0C, wd[31:0]);
    axi_write(KV + 32'h10, wd[63:32]);
    axi_write(KV + 32'h04, op | (s << 4) | (src << 8) | (n << 16));
    do axi_read(KV + 32'h14, st); while (!st[1]);
    axi_read(KV + 32'h18, lo);
    axi_read(KV + 32'h1C, hi);
    axi_read(KV + 32'h2C, cy);
    code = st[5:4]; val = {hi, lo}; last_cycles = cy;
  endtask

  function automatic logic [63:0] kvdata(input int s, input int t);
    return {16'hA9E5, 8'(s), 8'h00, 32'(t)};
  endfunction

  // scoreboard
  integer ntok [KS]; integer npg [KS]; integer nsh [KS]; bit dirty [KS][KP]; bit hasd [KS][KP];
  integer park_cyc, park_tok, rest_cyc, rest_tok;

  task automatic append(input int s, input int k);
    int code; logic [63:0] v;
    for (int i = 0; i < k; i++) begin
      int t;
      t = ntok[s];
      if (t % PT == 0) begin
        kv_cmd(ALLOC, s, 0, 0, 0, 0, code, v); check("alloc", code, 0);
        dirty[s][npg[s]] = 1; hasd[s][npg[s]] = 0; npg[s]++;
      end
      kv_cmd(WRITE, s, 0, 0, t, kvdata(s, t), code, v); check("write", code, 0);
      dirty[s][t / PT] = 1; ntok[s]++;
    end
  endtask

  task automatic verify(input int s);
    int code; logic [63:0] v;
    for (int t = 0; t < ntok[s]; t++) begin
      kv_cmd(READ, s, 0, 0, t, 0, code, v);
      check("read status", code, 0);
      check($sformatf("data s%0d t%0d", s, t), v, kvdata(t / PT < nsh[s] ? 0 : s, t));
    end
  endtask

  task automatic park(input int s);
    int code, want; logic [63:0] v;
    want = 0;
    for (int p = nsh[s]; p < npg[s]; p++) if (dirty[s][p] || !hasd[s][p]) want += PT;
    kv_cmd(PARK, s, 0, 0, 0, 0, code, v);
    check("park status", code, 0);
    check($sformatf("park s%0d tokens", s), v, want);
    park_cyc += last_cycles; park_tok += want;
    for (int p = nsh[s]; p < npg[s]; p++) begin dirty[s][p] = 0; hasd[s][p] = 1; end
  endtask

  task automatic restore(input int s);
    int code; logic [63:0] v;
    kv_cmd(RESTORE, s, 0, 0, 0, 0, code, v);
    check("restore status", code, 0);
    check($sformatf("restore s%0d tokens", s), v, (npg[s] - nsh[s]) * PT);
    rest_cyc += last_cycles; rest_tok += (npg[s] - nsh[s]) * PT;
  endtask

  // ------------------------------------------------------------------ test
  initial begin
    logic [31:0] v, cycles, s1, sl;
    int code; logic [63:0] v64;
    errors = 0; checks = 0; park_cyc = 0; park_tok = 0; rest_cyc = 0; rest_tok = 0;
    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0; awaddr = 0; wdata = 0; araddr = 0;
    for (int s = 0; s < KS; s++) begin
      ntok[s] = 0; npg[s] = 0; nsh[s] = 0;
      for (int p = 0; p < KP; p++) begin dirty[s][p] = 0; hasd[s][p] = 0; end
    end
    $readmemh({`VEC_DIR, "/weights.hex"}, wmem);
    $readmemh({`VEC_DIR, "/acts.hex"}, xmem);
    $readmemh({`VEC_DIR, "/expected.hex"}, ymem);
    repeat (4) @(posedge clk);
    rst_n = 1;

    // ---- 1. routing: both blocks answer at their own addresses
    axi_read(32'h0, v);      check("tile ID", v, 32'h1A1C_0001);
    axi_read(KV, v);         check("KV ID", v, 32'h1A1C_4B56);
    axi_read(KV + 32'h34, v); check("KV tiers", v, KH | (KD << 16));

    // ---- 2. tile matmul through the router
    axi_write(32'hC, NB);
    axi_write(32'h10, M);
    for (int r = 0; r < NB * N; r++)
      for (int w = 0; w < WPR; w++)
        axi_write(32'h10_0000 + (r * WPR + w) * 4,
                  {wmem[r*N+4*w+3], wmem[r*N+4*w+2], wmem[r*N+4*w+1], wmem[r*N+4*w]});
    for (int r = 0; r < NB * M; r++)
      for (int w = 0; w < WPR; w++)
        axi_write(32'h20_0000 + (r * WPR + w) * 4,
                  {xmem[r*N+4*w+3], xmem[r*N+4*w+2], xmem[r*N+4*w+1], xmem[r*N+4*w]});
    axi_write(32'h4, 1);
    do axi_read(32'h8, v); while (!v[1]);
    for (int r = 0; r < NB * M; r++)
      for (int n = 0; n < N; n++) begin
        axi_read(32'h30_0000 + (r * N + n) * 4, v);
        check($sformatf("tile out r%0d c%0d", r, n), v, ymem[r*N + n]);
      end
    axi_read(32'h1C, s1); axi_read(32'h20, sl);
    check("tile cycles/block", (sl - s1) / (NB - 2), (M > N / LANES) ? M : N / LANES);

    // ---- 3. agents through the KV registers
    append(0, PREFIX * PT);                                  // shared system prompt
    for (int a = 1; a <= AGENTS; a++) begin
      kv_cmd(SHARE, a, 0, PREFIX, 0, 0, code, v64); check("share", code, 0);
      npg[a] = PREFIX; ntok[a] = PREFIX * PT; nsh[a] = PREFIX;
    end
    kv_cmd(WRITE, 2, 0, 0, 3, 64'h1, code, v64); check("shared prompt is read-only", code, 3);
    for (int turn = 0; turn < 3; turn++)
      for (int a = 1; a <= AGENTS; a++) begin
        if (npg[a] > nsh[a]) restore(a);
        append(a, 12 + 7 * turn);
        verify(a);
        park(a);
      end
    kv_cmd(READ, 1, 0, 0, ntok[1] - 1, 0, code, v64); check("parked page is not readable", code, 3);
    for (int a = AGENTS; a >= 0; a--) begin kv_cmd(FREE, a, 0, 0, 0, 0, code, v64); check("free", code, 0); end
    axi_read(KV + 32'h28, v); check("no leaked pages", v, KH | (KD << 16));

    $display("RESULT ia1_top N=%0d | tile matmul + KV agents through one AXI-Lite port | park: %0d tokens in %0d cycles (%.2f cycles/token) | restore: %0d tokens in %0d cycles (%.2f cycles/token) | checks=%0d errors=%0d | %s",
             N, park_tok, park_cyc, $itor(park_cyc) / park_tok, rest_tok, rest_cyc, $itor(rest_cyc) / rest_tok,
             checks, errors, errors == 0 ? "PASS" : "FAIL");
    $finish;
  end

  initial begin #(80_000_000); $display("RESULT TIMEOUT"); $finish; end
endmodule
