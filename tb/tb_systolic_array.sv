// Self-checking testbench for systolic_array.
//
// Plays the role of the tile sequencer: streams NB weight blocks of M rows
// each, loading the next blocks' weights through the wavefront lanes while the
// current block streams. Checks every output row against the Python golden
// model and measures the steady-state cycles per block.
//
// +EARLY=k lets a lane start while the block two back still has k rows left to
// present. k = 0 is the legal schedule; k >= 1 violates the contract and
// should produce mismatches (proves the timing window is tight, not padded).
`timescale 1ns/1ps
module tb_systolic_array;
  `include "params.svh"
  localparam int AW   = 8;
  localparam int ACCW = 32;
  localparam int RW   = $clog2(N);

  logic                  clk = 1'b0;
  logic                  rst_n = 1'b0;
  logic                  in_valid, in_buf;
  logic [N*AW-1:0]       in_act;
  logic                  out_valid;
  logic [N*ACCW-1:0]     out_psum;
  logic [LANES-1:0]      wl_valid, wl_buf;
  logic [LANES*RW-1:0]   wl_row;
  logic [LANES*N*AW-1:0] wl_data;

  always #5 clk = ~clk;

  systolic_array #(.N(N), .AW(AW), .ACCW(ACCW), .LANES(LANES)) dut (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid), .in_buf(in_buf), .in_act(in_act),
    .out_valid(out_valid), .out_psum(out_psum),
    .wl_valid(wl_valid), .wl_buf(wl_buf), .wl_row(wl_row), .wl_data(wl_data)
  );

  logic [7:0]  wmem [0:NB*N*N-1];
  logic [7:0]  xmem [0:NB*M*N-1];
  logic [31:0] ymem [0:NB*M*N-1];

  // sequencer state
  integer lane_active [0:LANES-1];
  integer lane_block  [0:LANES-1];
  integer lane_row    [0:LANES-1];
  integer load_start  [0:NB-1];   // cycle the block's lane presented row 0
  integer issue_start [0:NB-1];   // cycle the block's first row was presented
  integer issued      [0:NB-1];   // rows presented so far
  integer cyc, next_load, next_issue, early, stalls, outs, errors;

  // ------------------------------------------------------------- checker
  always @(negedge clk) begin
    if (rst_n && out_valid) begin
      for (int n = 0; n < N; n++) begin
        if (out_psum[n*ACCW +: ACCW] !== ymem[outs*N + n]) begin
          errors = errors + 1;
          if (errors <= 5)
            $display("MISMATCH row %0d (block %0d) col %0d: got %0d expected %0d", outs, outs / M, n,
                     $signed(out_psum[n*ACCW +: ACCW]), $signed(ymem[outs*N + n]));
        end
      end
      outs = outs + 1;
    end
  end

  // ----------------------------------------------------------- sequencer
  initial begin
    real period;
    integer model_period;
    $readmemh("build/weights.hex", wmem);
    $readmemh("build/acts.hex", xmem);
    $readmemh("build/expected.hex", ymem);
    if (!$value$plusargs("EARLY=%d", early)) early = 0;

    for (int i = 0; i < LANES; i++) lane_active[i] = 0;
    for (int i = 0; i < NB; i++) begin load_start[i] = -1; issue_start[i] = -1; issued[i] = 0; end
    cyc = 0; next_load = 0; next_issue = 0; stalls = 0; outs = 0; errors = 0;
    in_valid = 0; in_buf = 0; in_act = '0;
    wl_valid = '0; wl_buf = '0; wl_row = '0; wl_data = '0;

    repeat (3) @(posedge clk);
    rst_n = 1'b1;

    while (next_issue < NB) begin
      @(negedge clk);

      // 1. start a lane for the next block once its buffer's previous user
      //    (two blocks back) has presented all but EARLY of its rows
      for (int l = 0; l < LANES; l++) begin
        if (!lane_active[l] && next_load < NB &&
            (next_load < 2 || issued[next_load-2] >= M - early)) begin
          lane_active[l] = 1;
          lane_block[l]  = next_load;
          lane_row[l]    = 0;
          load_start[next_load] = cyc;
          next_load = next_load + 1;
        end
      end

      // 2. every active lane writes one row this cycle, rows in order
      wl_valid = '0;
      for (int l = 0; l < LANES; l++) begin
        if (lane_active[l]) begin
          wl_valid[l] = 1'b1;
          wl_buf[l]   = lane_block[l] % 2;
          wl_row[l*RW +: RW] = lane_row[l];
          for (int n = 0; n < N; n++)
            wl_data[(l*N + n)*AW +: AW] = wmem[(lane_block[l]*N + lane_row[l])*N + n];
          lane_row[l] = lane_row[l] + 1;
          if (lane_row[l] == N) lane_active[l] = 0;
        end
      end

      // 3. present the next activation row once its block's lane has started
      if (load_start[next_issue] >= 0 && cyc >= load_start[next_issue] + 1) begin
        if (issued[next_issue] == 0) issue_start[next_issue] = cyc;
        in_valid = 1'b1;
        in_buf   = next_issue % 2;
        for (int k = 0; k < N; k++)
          in_act[k*AW +: AW] = xmem[(next_issue*M + issued[next_issue])*N + k];
        issued[next_issue] = issued[next_issue] + 1;
        if (issued[next_issue] == M) next_issue = next_issue + 1;
      end else begin
        in_valid = 1'b0;
        stalls = stalls + 1;
      end
      cyc = cyc + 1;
    end

    @(negedge clk);
    in_valid = 1'b0;
    wl_valid = '0;

    // drain: every row leaves 2N-1 cycles after it entered
    repeat (2 * N + 4) @(negedge clk);

    period = $itor(issue_start[NB-1] - issue_start[1]) / (NB - 2);
    model_period = (M > (N + LANES - 1) / LANES) ? M : (N + LANES - 1) / LANES;
    if (outs != NB * M) begin
      errors = errors + 1;
      $display("MISSING outputs: got %0d rows, expected %0d", outs, NB * M);
    end
    $display("RESULT N=%0d LANES=%0d M=%0d blocks=%0d early=%0d | cycles/block measured=%0.2f model=max(M,N/LANES)=%0d | stall cycles=%0d | errors=%0d | %s",
             N, LANES, M, NB, early, period, model_period, stalls, errors, errors == 0 ? "PASS" : "FAIL");
    $finish;
  end

  initial begin
    #(200000 * 10);
    $display("RESULT TIMEOUT");
    $finish;
  end
endmodule
