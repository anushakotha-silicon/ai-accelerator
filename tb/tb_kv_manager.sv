// Multi-agent test of kv_manager with a cycle-free scoreboard.
//
// Session 0 holds a shared system prompt; sessions 1..S-1 are agents that map
// it with SHARE, then run turns: restore, append tool output + generated
// tokens, read back every token, park. HBM is deliberately too small for all
// agents at once, so parking is required. The scoreboard predicts exactly how
// many tokens each PARK/RESTORE moves (incremental parking copies only dirty
// pages) and checks every token's data after every round trip.
`timescale 1ns/1ps
module tb_kv_manager;
  localparam int S = 8, P = 16, PAGE_TOK = 16, H = 32, D = 128, DATA_W = 64;   // capacity tier > fast tier
  localparam int PREFIX_PAGES = 2;
  localparam int SW = $clog2(S), PW = $clog2(P), TW = $clog2(P * PAGE_TOK);
  localparam logic [2:0] ALLOC = 1, WRITE = 2, READ = 3, PARK = 4, RESTORE = 5, FREE = 6, SHARE = 7;
  localparam logic [1:0] OK = 0, NOSPACE = 1, RANGE = 2, BADSTATE = 3;

  logic clk = 1'b0, rst_n = 1'b0;
  always #2 clk = ~clk;

  logic cmd_valid, cmd_ready, rsp_valid, rsp_ready;
  logic [2:0] cmd_op;
  logic [SW-1:0] cmd_sess, cmd_src;
  logic [PW:0] cmd_n;
  logic [TW-1:0] cmd_tok;
  logic [DATA_W-1:0] cmd_wdata, rsp_value;
  logic [1:0] rsp_status;
  logic [31:0] parked, restored;
  logic [$clog2(H):0] hbm_free;
  logic [$clog2(D):0] ddr_free;

  kv_manager #(.S(S), .P(P), .PAGE_TOK(PAGE_TOK), .H(H), .D(D), .DATA_W(DATA_W)) dut (
    .clk(clk), .rst_n(rst_n),
    .cmd_valid(cmd_valid), .cmd_ready(cmd_ready), .cmd_op(cmd_op), .cmd_sess(cmd_sess),
    .cmd_src(cmd_src), .cmd_n(cmd_n), .cmd_tok(cmd_tok), .cmd_wdata(cmd_wdata),
    .rsp_valid(rsp_valid), .rsp_ready(rsp_ready), .rsp_status(rsp_status), .rsp_value(rsp_value),
    .stat_tok_parked(parked), .stat_tok_restored(restored),
    .stat_hbm_free(hbm_free), .stat_ddr_free(ddr_free)
  );

  // ---------------------------------------------------------- scoreboard
  integer ntok     [S];          // tokens in the session (including shared prefix)
  integer npg      [S];
  integer nshared  [S];          // leading shared (read-only, pinned) pages
  bit     resident [S];
  bit     dirty    [S][P];
  bit     has_ddr  [S][P];
  integer errors, checks, naive_park_tok, parks;

  function automatic logic [DATA_W-1:0] kv(input int s, input int t);
    return {16'hA9E5, 8'(s), 8'h00, 32'(t)};      // unique per (session, token)
  endfunction

  // ------------------------------------------------------------ BFM
  task automatic cmd(input logic [2:0] op, input int s, input int src, input int n, input int tok,
                     input logic [DATA_W-1:0] wd, output logic [1:0] st, output logic [DATA_W-1:0] val);
    @(negedge clk);
    cmd_valid = 1; cmd_op = op; cmd_sess = SW'(s); cmd_src = SW'(src); cmd_n = (PW+1)'(n);
    cmd_tok = TW'(tok); cmd_wdata = wd;
    do @(posedge clk); while (!cmd_ready);
    @(negedge clk); cmd_valid = 0;
    do @(posedge clk); while (!rsp_valid);
    st = rsp_status; val = rsp_value;
    @(negedge clk); rsp_ready = 1;
    @(negedge clk); rsp_ready = 0;
  endtask

  task automatic expect_status(input string what, input logic [1:0] got, input logic [1:0] want);
    checks++;
    if (got !== want) begin
      errors++;
      if (errors <= 8) $display("ERROR %s: status %0d, expected %0d", what, got, want);
    end
  endtask

  task automatic expect_value(input string what, input logic [DATA_W-1:0] got, input logic [DATA_W-1:0] want);
    checks++;
    if (got !== want) begin
      errors++;
      if (errors <= 8) $display("ERROR %s: got %h, expected %h", what, got, want);
    end
  endtask

  // append k tokens to session s (allocating pages as needed)
  task automatic append(input int s, input int k);
    logic [1:0] st; logic [DATA_W-1:0] v;
    for (int i = 0; i < k; i++) begin
      int t = ntok[s];
      if (t % PAGE_TOK == 0) begin
        cmd(ALLOC, s, 0, 0, 0, '0, st, v);
        expect_status($sformatf("alloc s%0d", s), st, OK);
        dirty[s][npg[s]] = 1; has_ddr[s][npg[s]] = 0;
        npg[s]++;
      end
      cmd(WRITE, s, 0, 0, t, kv(s, t), st, v);
      expect_status($sformatf("write s%0d t%0d", s, t), st, OK);
      dirty[s][t / PAGE_TOK] = 1;
      ntok[s]++;
    end
  endtask

  // read back every token of session s; shared prefix must read as session 0's data
  task automatic verify(input int s);
    logic [1:0] st; logic [DATA_W-1:0] v;
    for (int t = 0; t < ntok[s]; t++) begin
      cmd(READ, s, 0, 0, t, '0, st, v);
      expect_status($sformatf("read s%0d t%0d", s, t), st, OK);
      expect_value($sformatf("data s%0d t%0d", s, t), v,
                   kv(t / PAGE_TOK < nshared[s] ? 0 : s, t));
    end
  endtask

  task automatic park(input int s);
    logic [1:0] st; logic [DATA_W-1:0] v; int want = 0;
    for (int p = nshared[s]; p < npg[s]; p++) if (dirty[s][p] || !has_ddr[s][p]) want += PAGE_TOK;
    naive_park_tok += (npg[s] - nshared[s]) * PAGE_TOK;       // what a non-incremental design copies
    parks++;
    cmd(PARK, s, 0, 0, 0, '0, st, v);
    expect_status($sformatf("park s%0d", s), st, OK);
    expect_value($sformatf("park s%0d tokens moved", s), v, DATA_W'(want));
    for (int p = nshared[s]; p < npg[s]; p++) begin dirty[s][p] = 0; has_ddr[s][p] = 1; end
    resident[s] = 0;
  endtask

  task automatic restore(input int s);
    logic [1:0] st; logic [DATA_W-1:0] v;
    cmd(RESTORE, s, 0, 0, 0, '0, st, v);
    expect_status($sformatf("restore s%0d", s), st, OK);
    expect_value($sformatf("restore s%0d tokens moved", s), v, DATA_W'((npg[s] - nshared[s]) * PAGE_TOK));
    resident[s] = 1;
  endtask

  // -------------------------------------------------------------- test
  initial begin
    logic [1:0] st; logic [DATA_W-1:0] v;
    int fit, seed, s, k;
    bit ok;
    errors = 0; checks = 0; naive_park_tok = 0; parks = 0;
    cmd_valid = 0; rsp_ready = 0; cmd_op = 0; cmd_sess = 0; cmd_src = 0; cmd_n = 0; cmd_tok = 0; cmd_wdata = 0;
    for (int i = 0; i < S; i++) begin
      ntok[i] = 0; npg[i] = 0; nshared[i] = 0; resident[i] = 1;
      for (int p = 0; p < P; p++) begin dirty[i][p] = 0; has_ddr[i][p] = 0; end
    end
    repeat (4) @(posedge clk);
    rst_n = 1;

    // ---- 1. shared system prompt: written once, mapped into 7 agents
    append(0, PREFIX_PAGES * PAGE_TOK);
    for (int a = 1; a < S; a++) begin
      cmd(SHARE, a, 0, PREFIX_PAGES, 0, '0, st, v);
      expect_status($sformatf("share into s%0d", a), st, OK);
      npg[a] = PREFIX_PAGES; ntok[a] = PREFIX_PAGES * PAGE_TOK; nshared[a] = PREFIX_PAGES;
    end
    expect_value("HBM pages used by prompt + 7 shares", DATA_W'(H - hbm_free), DATA_W'(PREFIX_PAGES));
    cmd(WRITE, 3, 0, 0, 5, 64'hBAD, st, v);
    expect_status("write into shared prefix is refused", st, BADSTATE);
    verify(5);

    // ---- 2. capacity: grow every agent to 6 private pages with no parking
    fit = 0;
    for (int a = 1; a < S; a++) begin
      ok = 1;
      for (int i = 0; i < 6 && ok; i++) begin
        cmd(ALLOC, a, 0, 0, 0, '0, st, v);
        if (st == NOSPACE) ok = 0;
        else begin dirty[a][npg[a]] = 1; has_ddr[a][npg[a]] = 0; npg[a]++; end
      end
      // fill allocated pages so the scoreboard's token count matches
      for (int t = ntok[a]; t < npg[a] * PAGE_TOK; t++) begin
        cmd(WRITE, a, 0, 0, t, kv(a, t), st, v);
        expect_status("fill", st, OK);
      end
      ntok[a] = npg[a] * PAGE_TOK;
      if (ok) fit++;
      else begin
        $display("CAPACITY: HBM full after %0d agents x 6 private pages (+ one shared prompt); agent %0d gets NOSPACE", fit, a);
        break;
      end
    end
    // park everyone so the turn phase starts from a clean state
    for (int a = 1; a < S; a++) if (npg[a] > nshared[a]) park(a); else resident[a] = 0;
    expect_value("HBM after parking all agents: only the prompt", DATA_W'(H - hbm_free), DATA_W'(PREFIX_PAGES));

    // ---- 3. agent turns: restore, append tool output + generation, verify, park
    for (int turn = 0; turn < 3; turn++)
      for (int a = 1; a < S; a++) begin
        if (npg[a] > nshared[a]) restore(a); else resident[a] = 1;
        append(a, 9 + 5 * turn);           // tool output + generated tokens this turn
        verify(a);
        park(a);
      end

    // ---- 4. random stress: random agent, random append, occasional read of a parked agent
    seed = 7;
    for (int i = 0; i < 40; i++) begin
      s = 1 + $urandom(seed) % (S - 1); seed = seed + 1;
      k = 1 + $urandom(seed) % 12;      seed = seed + 1;
      if (($urandom(seed) % 4) == 0 && !resident[s] && ntok[s] > nshared[s] * PAGE_TOK) begin
        cmd(READ, s, 0, 0, ntok[s] - 1, '0, st, v);
        expect_status("read of a parked private page is refused", st, BADSTATE);
      end
      seed = seed + 1;
      if (ntok[s] + k > P * PAGE_TOK) continue;
      if (!resident[s]) begin
        if (npg[s] > nshared[s]) restore(s); else resident[s] = 1;
      end
      append(s, k);
      verify(s);
      park(s);
    end

    // ---- 5. free everything: no page may leak
    for (int a = S - 1; a >= 0; a--) begin
      cmd(FREE, a, 0, 0, 0, '0, st, v);
      expect_status($sformatf("free s%0d", a), st, OK);
    end
    @(negedge clk);
    expect_value("HBM pages free at end", DATA_W'(hbm_free), DATA_W'(H));
    expect_value("DDR pages free at end", DATA_W'(ddr_free), DATA_W'(D));

    $display("RESULT kv_manager | agents fitting in HBM without parking: %0d of %0d | parks=%0d tokens parked=%0d (a non-incremental design: %0d, %.2fx more) | restored=%0d | checks=%0d errors=%0d | %s",
             fit, S - 1, parks, parked, naive_park_tok, $itor(naive_park_tok) / parked, restored, checks, errors,
             errors == 0 ? "PASS" : "FAIL");
    $finish;
  end

  initial begin #(20_000_000); $display("RESULT TIMEOUT"); $finish; end
endmodule
