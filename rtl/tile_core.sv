// FPGA-facing tile core: systolic array + on-chip buffers + hardware
// sequencer + AXI4-Lite control. Maps onto the AWS F2 OCL (BAR0) port.
//
// Host flow: write NUM_BLOCKS and ROWS_PER_BLOCK, fill the weight and
// activation buffers, write CTRL.start, poll STATUS.done, read the output
// buffer and the cycle counters.
//
// Register map (byte addresses, 32-bit words; wstrb ignored, full words only)
//   0x000000  ID              RO  0x1A1C_0001
//   0x000004  CTRL            WO  bit0 = start (ignored while busy)
//   0x000008  STATUS          RO  bit0 = busy, bit1 = done
//   0x00000C  NUM_BLOCKS      RW  1..MAX_BLOCKS
//   0x000010  ROWS_PER_BLOCK  RW  1..MAX_M (M)
//   0x000014  CYCLES          RO  start to last output row
//   0x000018  STALL_CYCLES    RO  cycles with rows left but none presented
//   0x00001C  BLK1_START      RO  cycle block 1 presented its first row
//   0x000020  LAST_START      RO  cycle the last block presented its first row
//   0x000024  PARAMS          RO  N | LANES<<8 | MAX_BLOCKS<<16 | MAX_M<<24
//   0x100000  weight buffer   RW  row (b*N + k), element n in byte n of the row
//   0x200000  activation buf  RW  row (b*M + m), element k in byte k of the row
//   0x300000  output buffer   RO  row (b*M + m), element n = word n (INT32)
//
// Steady-state cycles/block = (LAST_START - BLK1_START) / (NUM_BLOCKS - 2),
// which the model predicts as max(M, N / LANES).
module tile_core #(
  parameter int N          = 16,   // power of two, >= 8
  parameter int LANES      = 2,
  parameter int AW         = 8,
  parameter int ACCW       = 32,
  parameter int MAX_BLOCKS = 16,   // power of two
  parameter int MAX_M      = 64
) (
  input  logic        clk,
  input  logic        rst_n,
  input  logic [31:0] s_awaddr,
  input  logic        s_awvalid,
  output logic        s_awready,
  input  logic [31:0] s_wdata,
  input  logic        s_wvalid,
  output logic        s_wready,
  output logic [1:0]  s_bresp,
  output logic        s_bvalid,
  input  logic        s_bready,
  input  logic [31:0] s_araddr,
  input  logic        s_arvalid,
  output logic        s_arready,
  output logic [31:0] s_rdata,
  output logic [1:0]  s_rresp,
  output logic        s_rvalid,
  input  logic        s_rready
);
  localparam int RW    = $clog2(N);
  localparam int KW    = $clog2(MAX_BLOCKS);          // block index into buffers
  localparam int BW    = $clog2(MAX_BLOCKS + 1);      // block counters (0..MAX_BLOCKS)
  localparam int MW    = $clog2(MAX_M + 1);
  localparam int WPR   = N * AW / 32;                 // 32-bit words per weight/activation row
  localparam int LWPR  = $clog2(WPR);
  localparam int LN    = $clog2(N);                   // words per output row = N
  localparam int WROWS = MAX_BLOCKS * N;
  localparam int AROWS = MAX_BLOCKS * MAX_M;
  localparam int WAW   = $clog2(WROWS);
  localparam int AAW   = $clog2(AROWS);

  // ------------------------------------------------------------ buffers
  logic [N*AW-1:0]   wmem [WROWS];
  logic [N*AW-1:0]   amem [AROWS];
  logic [N*ACCW-1:0] omem [AROWS];

  // ---------------------------------------------------------- registers
  logic [BW-1:0] reg_nb;
  logic [MW-1:0] reg_m;
  logic          busy, done;
  logic [31:0]   cyc_cnt, stall_cnt, blk1_start, last_start;

  // ------------------------------------------------- AXI-Lite write path
  logic        aw_hold, w_hold;
  // Only address bits [23:0] are decoded; the rest of the 64 MiB BAR aliases.
  /* verilator lint_off UNUSEDSIGNAL */
  logic [31:0] aw_q, w_q;
  logic        host_we;
  logic        start_req;

  assign s_awready = !aw_hold;
  assign s_wready  = !w_hold;
  assign s_bresp   = 2'b00;
  assign host_we   = aw_hold && w_hold && !s_bvalid;
  assign start_req = host_we && aw_q[23:20] == 4'h0 && aw_q[19:0] == 20'h4 && w_q[0] && !busy;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      aw_hold <= 1'b0; w_hold <= 1'b0; s_bvalid <= 1'b0; aw_q <= '0; w_q <= '0;
      reg_nb <= BW'(1); reg_m <= MW'(1);
    end else begin
      if (s_awvalid && s_awready) begin aw_hold <= 1'b1; aw_q <= s_awaddr; end
      if (s_wvalid && s_wready)   begin w_hold  <= 1'b1; w_q  <= s_wdata;  end
      if (host_we) begin
        aw_hold  <= 1'b0;
        w_hold   <= 1'b0;
        s_bvalid <= 1'b1;
        if (aw_q[23:20] == 4'h0 && !busy) begin
          if (aw_q[19:0] == 20'hC) reg_nb <= w_q[BW-1:0];
          if (aw_q[19:0] == 20'h10) reg_m <= w_q[MW-1:0];
        end
      end
      if (s_bvalid && s_bready) s_bvalid <= 1'b0;
    end
  end

  // host writes into the input buffers (only while idle)
  wire [17:0] host_word = aw_q[19:2];
  /* verilator lint_on UNUSEDSIGNAL */
  always_ff @(posedge clk) begin
    if (host_we && !busy && aw_q[23:20] == 4'h1)
      wmem[host_word[LWPR +: WAW]][host_word[LWPR-1:0]*32 +: 32] <= w_q;
    if (host_we && !busy && aw_q[23:20] == 4'h2)
      amem[host_word[LWPR +: AAW]][host_word[LWPR-1:0]*32 +: 32] <= w_q;
  end

  // -------------------------------------------------- AXI-Lite read path
  logic        rd_pend;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [31:0] ar_q;
  wire  [17:0] rd_word = ar_q[19:2];
  /* verilator lint_on UNUSEDSIGNAL */
  assign s_arready = !rd_pend && !s_rvalid;
  assign s_rresp   = 2'b00;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_pend <= 1'b0; s_rvalid <= 1'b0; s_rdata <= '0; ar_q <= '0;
    end else begin
      if (s_arvalid && s_arready) begin rd_pend <= 1'b1; ar_q <= s_araddr; end
      if (rd_pend) begin
        rd_pend  <= 1'b0;
        s_rvalid <= 1'b1;
        case (ar_q[23:20])
          4'h0: case (ar_q[19:0])
                  20'h00:  s_rdata <= 32'h1A1C_0001;
                  20'h08:  s_rdata <= {30'd0, done, busy};
                  20'h0C:  s_rdata <= 32'(reg_nb);
                  20'h10:  s_rdata <= 32'(reg_m);
                  20'h14:  s_rdata <= cyc_cnt;
                  20'h18:  s_rdata <= stall_cnt;
                  20'h1C:  s_rdata <= blk1_start;
                  20'h20:  s_rdata <= last_start;
                  20'h24:  s_rdata <= 32'(N) | (32'(LANES) << 8) | (32'(MAX_BLOCKS) << 16) | (32'(MAX_M) << 24);
                  default: s_rdata <= 32'hDEAD_BEEF;
                endcase
          4'h1:    s_rdata <= wmem[rd_word[LWPR +: WAW]][rd_word[LWPR-1:0]*32 +: 32];
          4'h2:    s_rdata <= amem[rd_word[LWPR +: AAW]][rd_word[LWPR-1:0]*32 +: 32];
          4'h3:    s_rdata <= omem[rd_word[LN +: AAW]][rd_word[LN-1:0]*ACCW +: 32];
          default: s_rdata <= 32'hDEAD_BEEF;
        endcase
      end
      if (s_rvalid && s_rready) s_rvalid <= 1'b0;
    end
  end

  // ------------------------------------------------ hardware sequencer
  // Same schedule the testbench proved: (1) present the next activation row if
  // its block's lane started on an earlier cycle; (2) start at most one lane,
  // for the next block, once the block two back has presented its last row
  // (this cycle counts: s0 = L); (3) every active lane writes one row.
  logic [BW-1:0]  next_issue, next_load;
  logic [MW-1:0]  row_ctr;
  logic [AAW:0]   act_ptr, out_ptr, total_rows;
  wire  [AAW:0]   nb_ext = {{(AAW + 1 - BW){1'b0}}, reg_nb};
  wire  [AAW:0]   m_ext  = {{(AAW + 1 - MW){1'b0}}, reg_m};
  logic [LANES-1:0] lane_active;
  logic [KW-1:0]  lane_block [LANES];
  logic [RW-1:0]  lane_row   [LANES];

  logic issue_ok, last_row, prev_done, start_ok;
  logic [LANES-1:0] lane_drv, lane_start;
  logic [KW-1:0]  drv_block [LANES];
  logic [RW-1:0]  drv_row   [LANES];

  always_comb begin
    issue_ok = busy && next_issue < reg_nb && next_issue < next_load;
    last_row = issue_ok && row_ctr == reg_m - 1'b1;
    // has block next_load-2 presented all its rows (including this cycle)?
    prev_done = next_load < 2 ||
                next_issue > next_load - BW'(2) ||
                (next_issue == next_load - BW'(2) && last_row);
    start_ok = busy && next_load < reg_nb && prev_done && !(&lane_active);
    lane_start = '0;
    for (int l = LANES - 1; l >= 0; l--)       // lowest free lane wins
      if (start_ok && !lane_active[l]) lane_start = LANES'(1) << l;
    for (int l = 0; l < LANES; l++) begin
      lane_drv[l]  = lane_active[l] || lane_start[l];
      drv_block[l] = lane_start[l] ? next_load[KW-1:0] : lane_block[l];
      drv_row[l]   = lane_start[l] ? '0 : lane_row[l];
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0; done <= 1'b0;
      next_issue <= '0; next_load <= '0; row_ctr <= '0; act_ptr <= '0; total_rows <= '0;
      lane_active <= '0;
      cyc_cnt <= '0; stall_cnt <= '0; blk1_start <= '0; last_start <= '0;
      for (int l = 0; l < LANES; l++) begin lane_block[l] <= '0; lane_row[l] <= '0; end
    end else if (start_req) begin
      busy <= 1'b1; done <= 1'b0;
      next_issue <= '0; next_load <= '0; row_ctr <= '0; act_ptr <= '0;
      total_rows <= nb_ext * m_ext;
      lane_active <= '0;
      cyc_cnt <= '0; stall_cnt <= '0; blk1_start <= '0; last_start <= '0;
    end else if (busy) begin
      cyc_cnt <= cyc_cnt + 1;
      if (next_issue < reg_nb && !issue_ok) stall_cnt <= stall_cnt + 1;
      if (issue_ok) begin
        act_ptr <= act_ptr + 1'b1;
        if (row_ctr == 0 && next_issue == 1) blk1_start <= cyc_cnt;
        if (row_ctr == 0 && next_issue == reg_nb - 1'b1) last_start <= cyc_cnt;
        row_ctr <= last_row ? '0 : row_ctr + 1'b1;
        if (last_row) next_issue <= next_issue + 1'b1;
      end
      if (start_ok) next_load <= next_load + 1'b1;
      for (int l = 0; l < LANES; l++) begin
        if (lane_drv[l]) begin
          lane_block[l]  <= drv_block[l];
          lane_row[l]    <= drv_row[l] + 1'b1;
          lane_active[l] <= drv_row[l] != RW'(N - 1);
        end
      end
      if (out_ptr == total_rows) begin busy <= 1'b0; done <= 1'b1; end
    end
  end

  // ---------------------- stage 1: buffer reads feed the array (registered)
  logic                  in_valid_r, in_buf_r;
  logic [N*AW-1:0]       in_act_r;
  logic [LANES-1:0]      wl_valid_r, wl_buf_r;
  logic [LANES*RW-1:0]   wl_row_r;
  logic [LANES*N*AW-1:0] wl_data_r;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      in_valid_r <= 1'b0; in_buf_r <= 1'b0; wl_valid_r <= '0; wl_buf_r <= '0; wl_row_r <= '0;
    end else begin
      in_valid_r <= issue_ok;
      in_buf_r   <= next_issue[0];
      for (int l = 0; l < LANES; l++) begin
        wl_valid_r[l]            <= lane_drv[l] && busy;
        wl_buf_r[l]              <= drv_block[l][0];
        wl_row_r[l*RW +: RW]     <= drv_row[l];
      end
    end
  end
  always_ff @(posedge clk) begin
    in_act_r <= amem[act_ptr[AAW-1:0]];
    for (int l = 0; l < LANES; l++)
      wl_data_r[l*N*AW +: N*AW] <= wmem[{drv_block[l], drv_row[l]}];
  end

  // ------------------------------------------------------------- array
  logic              out_valid;
  logic [N*ACCW-1:0] out_psum;

  systolic_array #(.N(N), .AW(AW), .ACCW(ACCW), .LANES(LANES)) u_array (
    .clk(clk), .rst_n(rst_n),
    .in_valid(in_valid_r), .in_buf(in_buf_r), .in_act(in_act_r),
    .out_valid(out_valid), .out_psum(out_psum),
    .wl_valid(wl_valid_r), .wl_buf(wl_buf_r), .wl_row(wl_row_r), .wl_data(wl_data_r)
  );

  // ------------------------------------------------------ output capture
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)          out_ptr <= '0;
    else if (start_req)  out_ptr <= '0;
    else if (out_valid)  out_ptr <= out_ptr + 1'b1;
  end
  always_ff @(posedge clk)
    if (out_valid) omem[out_ptr[AAW-1:0]] <= out_psum;
endmodule
