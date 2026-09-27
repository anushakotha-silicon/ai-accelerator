// N x N weight-stationary systolic array with wavefront weight loading.
//
// Computes Y[m][n] = sum_k X[m][k] * W[k][n] for a stream of rows X[m].
// PE[r][c] holds W[r][c]; row r receives activation element k = r.
//
// Timing (row m presented on in_* during cycle t):
//   PE[r][c] consumes it at the clock edge ending cycle t + r + c
//   out_psum carries Y[m] during cycle t + 2N - 1 (deskewed)
//
// Weight loading
// --------------
// The block that last used a buffer is still being read along a diagonal for
// 2N-2 cycles after its final row enters. Rather than wait for that drain,
// each load lane writes one row per cycle, and the array delays column c of
// every lane by c cycles. Row r of the new weights then lands in PE[r][c]
// exactly one wavefront behind the old data. With LANES lanes the array
// sustains one block per max(M, N / LANES) cycles, where M is rows per block.
//
// Sequencer contract (see tb/tb_systolic_array.sv):
//   * a lane loading buffer b starts at row 0 no earlier than the cycle the
//     last row using b (two blocks back) was presented, and walks rows
//     0..N-1 on consecutive cycles;
//   * the first row of the new block is presented at least one cycle after
//     the lane started;
//   * two lanes never target the same row in the same cycle.
module systolic_array #(
  parameter int N     = 8,
  parameter int AW    = 8,
  parameter int ACCW  = 32,
  parameter int LANES = 2,
  parameter int RW    = $clog2(N)
) (
  input  logic                  clk,
  input  logic                  rst_n,
  // activation stream: one K-vector per cycle, element r at [r*AW +: AW]
  input  logic                  in_valid,
  input  logic                  in_buf,
  input  logic [N*AW-1:0]       in_act,
  // results: one output row per cycle, element c at [c*ACCW +: ACCW]
  output logic                  out_valid,
  output logic [N*ACCW-1:0]     out_psum,
  // weight load lanes: lane l writes row wl_row[l] of buffer wl_buf[l]
  input  logic [LANES-1:0]      wl_valid,
  input  logic [LANES-1:0]      wl_buf,
  input  logic [LANES*RW-1:0]   wl_row,
  input  logic [LANES*N*AW-1:0] wl_data   // lane l, column c at [(l*N + c)*AW +: AW]
);
  localparam int SW = 2 + RW + AW;        // skewed lane word: valid, buf, row, data

  // ---------------------------------------------------------------- buses
  // horizontal: index r*(N+1) + c, c = 0..N (c = N is the right edge).
  // Activations leaving the right edge are dropped by design; only the last
  // row's valid is used (it times out_valid).
  /* verilator lint_off UNUSEDSIGNAL */
  logic [N*(N+1)-1:0]      h_valid, h_buf;
  logic [N*(N+1)*AW-1:0]   h_act;
  /* verilator lint_on UNUSEDSIGNAL */
  // vertical: index r*N + c, r = 0..N (r = 0 is the top, fed with zero)
  logic [(N+1)*N*ACCW-1:0] v_psum;
  // skewed weight lanes: index l*N + c
  logic [LANES*N-1:0]      lw_valid, lw_buf;
  logic [LANES*N*RW-1:0]   lw_row;
  logic [LANES*N*AW-1:0]   lw_data;

  assign v_psum[N*ACCW-1:0] = '0;

  genvar r, c, l;

  // ------------------------------------------------ input skew: row r waits r cycles
  generate
    for (r = 0; r < N; r++) begin : g_skew
      localparam int H = r * (N + 1);
      if (r == 0) begin : g_direct
        assign h_valid[H]           = in_valid;
        assign h_buf[H]             = in_buf;
        assign h_act[H*AW +: AW]    = in_act[0 +: AW];
      end else begin : g_delay
        logic [AW+1:0] pipe [0:r-1];
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            for (int i = 0; i < r; i++) pipe[i] <= '0;
          end else begin
            pipe[0] <= {in_valid, in_buf, in_act[r*AW +: AW]};
            for (int i = 1; i < r; i++) pipe[i] <= pipe[i-1];
          end
        end
        assign {h_valid[H], h_buf[H], h_act[H*AW +: AW]} = pipe[r-1];
      end
    end
  endgenerate

  // ------------------------------------ weight-lane skew: column c waits c cycles
  generate
    for (l = 0; l < LANES; l++) begin : g_lane
      for (c = 0; c < N; c++) begin : g_col
        localparam int I = l * N + c;
        logic [SW-1:0] head, tail;
        assign head = {wl_valid[l], wl_buf[l], wl_row[l*RW +: RW], wl_data[I*AW +: AW]};
        if (c == 0) begin : g_direct
          assign tail = head;
        end else begin : g_delay
          logic [SW-1:0] pipe [0:c-1];
          always_ff @(posedge clk or negedge rst_n) begin
            if (!rst_n) begin
              for (int i = 0; i < c; i++) pipe[i] <= '0;
            end else begin
              pipe[0] <= head;
              for (int i = 1; i < c; i++) pipe[i] <= pipe[i-1];
            end
          end
          assign tail = pipe[c-1];
        end
        assign lw_valid[I]         = tail[SW-1];
        assign lw_buf[I]           = tail[SW-2];
        assign lw_row[I*RW +: RW]  = tail[AW +: RW];
        assign lw_data[I*AW +: AW] = tail[0 +: AW];
      end
    end
  endgenerate

  // ------------------------------------------------------------- PE grid
  generate
    for (r = 0; r < N; r++) begin : g_row
      for (c = 0; c < N; c++) begin : g_col
        localparam int HI = r * (N + 1) + c;   // horizontal input index
        localparam int VI = r * N + c;         // vertical input index
        logic          we, wb;
        logic [AW-1:0] wd;

        // at most one lane targets this row in a given cycle (sequencer contract)
        always_comb begin
          we = 1'b0;
          wb = 1'b0;
          wd = '0;
          for (int k = 0; k < LANES; k++) begin
            if (lw_valid[k*N + c] && lw_row[(k*N + c)*RW +: RW] == r) begin
              we = 1'b1;
              wb = lw_buf[k*N + c];
              wd = lw_data[(k*N + c)*AW +: AW];
            end
          end
        end

        pe #(.AW(AW), .ACCW(ACCW)) u_pe (
          .clk         (clk),
          .rst_n       (rst_n),
          .act_valid_i (h_valid[HI]),
          .act_buf_i   (h_buf[HI]),
          .act_i       (h_act[HI*AW +: AW]),
          .psum_i      (v_psum[VI*ACCW +: ACCW]),
          .w_we        (we),
          .w_buf       (wb),
          .w_data      (wd),
          .act_valid_o (h_valid[HI+1]),
          .act_buf_o   (h_buf[HI+1]),
          .act_o       (h_act[(HI+1)*AW +: AW]),
          .psum_o      (v_psum[(VI+N)*ACCW +: ACCW])
        );
      end
    end
  endgenerate

  // -------------------------- output deskew: column c waits N-1-c cycles
  generate
    for (c = 0; c < N; c++) begin : g_deskew
      localparam int D  = N - 1 - c;
      localparam int BI = N * N + c;           // bottom edge of column c
      if (D == 0) begin : g_direct
        assign out_psum[c*ACCW +: ACCW] = v_psum[BI*ACCW +: ACCW];
      end else begin : g_delay
        logic [ACCW-1:0] pipe [0:D-1];
        always_ff @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin
            for (int i = 0; i < D; i++) pipe[i] <= '0;
          end else begin
            pipe[0] <= v_psum[BI*ACCW +: ACCW];
            for (int i = 1; i < D; i++) pipe[i] <= pipe[i-1];
          end
        end
        assign out_psum[c*ACCW +: ACCW] = pipe[D-1];
      end
    end
  endgenerate

  // the last PE's forwarded valid lines up with the deskewed row
  assign out_valid = h_valid[(N - 1) * (N + 1) + N];
endmodule
