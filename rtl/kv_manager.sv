// Agent KV-cache manager: paged KV, two memory tiers, park/restore, prefix sharing.
//
// Model of what the chip's KV/prefix manager + DMA do for agent sessions:
//   * each session's KV is a list of fixed-size pages (PAGE_TOK tokens each);
//     a page table maps (session, logical page) -> physical page and tier;
//   * PARK copies a session's private pages from the fast tier (HBM) to the
//     capacity tier (DDR) and frees the HBM pages; RESTORE brings them back;
//   * incremental parking: after a restore the DDR copy stays valid, so the
//     next PARK copies only pages written since (dirty), not the whole history;
//   * prefix sharing: SHARE maps another session's first pages (e.g. the
//     system prompt) into a new session. Shared pages are read-only, pinned in
//     HBM, and reference-counted so they are freed only by their last user.
//
// Each token is one DATA_W-bit word here, standing in for that token's KV line.
// HBM and DDR are local arrays; on the FPGA they become the card's HBM and DDR.
//
// Command port (one command at a time; wait for rsp_valid, then rsp_ready):
//   op  name     args              response value
//   1   ALLOC    sess              new HBM page number
//   2   WRITE    sess, tok, wdata  -
//   3   READ     sess, tok         token data
//   4   PARK     sess              tokens copied HBM -> DDR
//   5   RESTORE  sess              tokens copied DDR -> HBM
//   6   FREE     sess              -
//   7   SHARE    sess(dst), src, n -   (dst gets src's first n pages)
// Status: 0 OK, 1 NOSPACE (tier full), 2 RANGE (bad page/token), 3 STATE
// (not resident, write to a shared page, or bad SHARE preconditions).
module kv_manager #(
  parameter int S        = 8,    // sessions
  parameter int P        = 16,   // logical pages per session
  parameter int PAGE_TOK = 16,   // tokens per page (power of two)
  parameter int H        = 32,   // HBM physical pages
  parameter int D        = 64,   // DDR physical pages
  parameter int DATA_W   = 64
) (
  input  logic              clk,
  input  logic              rst_n,
  input  logic              cmd_valid,
  output logic              cmd_ready,
  input  logic [2:0]        cmd_op,
  input  logic [$clog2(S)-1:0] cmd_sess,
  input  logic [$clog2(S)-1:0] cmd_src,
  input  logic [$clog2(P):0]   cmd_n,
  input  logic [$clog2(P*PAGE_TOK)-1:0] cmd_tok,
  input  logic [DATA_W-1:0] cmd_wdata,
  output logic              rsp_valid,
  input  logic              rsp_ready,
  output logic [1:0]        rsp_status,
  output logic [DATA_W-1:0] rsp_value,
  // statistics
  output logic [31:0]       stat_tok_parked,
  output logic [31:0]       stat_tok_restored,
  output logic [$clog2(H):0] stat_hbm_free,
  output logic [$clog2(D):0] stat_ddr_free
);
  localparam int SW = $clog2(S), PW = $clog2(P), OW = $clog2(PAGE_TOK);
  localparam int HW = $clog2(H), DW = $clog2(D), RC = $clog2(S + 1);
  localparam logic [2:0] ALLOC = 3'd1, WRITE = 3'd2, READ = 3'd3, PARK = 3'd4,
                         RESTORE = 3'd5, FREE = 3'd6, SHARE = 3'd7;
  localparam logic [1:0] OK = 2'd0, NOSPACE = 2'd1, RANGE = 2'd2, BADSTATE = 2'd3;

  // ------------------------------------------------------------ state
  logic [DATA_W-1:0] hbm_mem [H * PAGE_TOK];
  logic [DATA_W-1:0] ddr_mem [D * PAGE_TOK];

  // page table entries, index {session, logical page}
  logic [HW-1:0] pte_hbm    [S * P];
  logic [DW-1:0] pte_ddr    [S * P];
  logic [S*P-1:0] pte_in_hbm, pte_in_ddr, pte_dirty, pte_shared;
  logic [PW:0]   npages [S];

  logic [H-1:0]  hbm_free;             // 1 = free
  logic [D-1:0]  ddr_free;
  logic [RC-1:0] refcnt [H];

  // ------------------------------------------------------ free-page search
  logic [HW-1:0] hbm_pick;  logic hbm_any;
  logic [DW-1:0] ddr_pick;  logic ddr_any;
  always_comb begin
    hbm_pick = '0; hbm_any = 1'b0;
    for (int i = H - 1; i >= 0; i--) if (hbm_free[i]) begin hbm_pick = HW'(i); hbm_any = 1'b1; end
    ddr_pick = '0; ddr_any = 1'b0;
    for (int i = D - 1; i >= 0; i--) if (ddr_free[i]) begin ddr_pick = DW'(i); ddr_any = 1'b1; end
  end

  // ------------------------------------------------------- command FSM
  typedef enum logic [2:0] {IDLE, EXEC, SCAN, COPY, READ2, RESP} state_t;
  state_t st;
  logic [2:0]        op;
  logic [SW-1:0]     sess, src;
  logic [PW:0]       n, p;              // p: page being scanned
  logic [PW-1:0]     tpage;
  logic [OW-1:0]     toff;
  logic [DATA_W-1:0] wdata;
  logic              copy_to_ddr;       // direction of the running page copy
  logic [OW-1:0]     w;                 // word within the page being copied
  logic [31:0]       moved;             // tokens copied by this PARK/RESTORE

  wire [SW+PW-1:0] ti  = {sess, tpage};          // entry addressed by WRITE/READ
  wire [SW+PW-1:0] pi  = {sess, p[PW-1:0]};      // entry being scanned
  wire [SW+PW-1:0] si  = {src,  p[PW-1:0]};      // SHARE source entry

  assign cmd_ready = (st == IDLE);

  task automatic respond(input logic [1:0] status, input logic [DATA_W-1:0] value);
    rsp_status <= status;
    rsp_value  <= value;
    rsp_valid  <= 1'b1;
    st         <= RESP;
  endtask

  // release one HBM page reference
  task automatic drop_hbm(input logic [HW-1:0] pg);
    if (refcnt[pg] == RC'(1)) hbm_free[pg] <= 1'b1;
    refcnt[pg] <= refcnt[pg] - 1'b1;
  endtask

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE; rsp_valid <= 1'b0; rsp_status <= OK; rsp_value <= '0;
      hbm_free <= '1; ddr_free <= '1;
      pte_in_hbm <= '0; pte_in_ddr <= '0; pte_dirty <= '0; pte_shared <= '0;
      for (int i = 0; i < S; i++) npages[i] <= '0;
      for (int i = 0; i < H; i++) refcnt[i] <= '0;
      stat_tok_parked <= '0; stat_tok_restored <= '0;
      op <= '0; sess <= '0; src <= '0; n <= '0; p <= '0; tpage <= '0; toff <= '0;
      wdata <= '0; copy_to_ddr <= 1'b0; w <= '0; moved <= '0;
    end else begin
      case (st)
        IDLE: if (cmd_valid) begin
          op <= cmd_op; sess <= cmd_sess; src <= cmd_src; n <= cmd_n;
          tpage <= cmd_tok[OW +: PW]; toff <= cmd_tok[OW-1:0]; wdata <= cmd_wdata;
          p <= '0; moved <= '0;
          st <= EXEC;
        end

        EXEC: case (op)
          ALLOC: begin
            if (npages[sess] == (PW+1)'(P)) respond(RANGE, '0);
            else if (!hbm_any)              respond(NOSPACE, '0);
            else begin
              pte_hbm[{sess, npages[sess][PW-1:0]}] <= hbm_pick;
              pte_in_hbm[{sess, npages[sess][PW-1:0]}] <= 1'b1;
              pte_in_ddr[{sess, npages[sess][PW-1:0]}] <= 1'b0;
              pte_dirty [{sess, npages[sess][PW-1:0]}] <= 1'b1;
              pte_shared[{sess, npages[sess][PW-1:0]}] <= 1'b0;
              hbm_free[hbm_pick] <= 1'b0;
              refcnt[hbm_pick]   <= RC'(1);
              npages[sess]       <= npages[sess] + 1'b1;
              respond(OK, DATA_W'(hbm_pick));
            end
          end
          WRITE, READ: begin
            if ({1'b0, tpage} >= npages[sess])            respond(RANGE, '0);
            else if (!pte_in_hbm[ti])                     respond(BADSTATE, '0);
            else if (op == WRITE && pte_shared[ti])       respond(BADSTATE, '0);
            else if (op == WRITE) begin
              hbm_mem[{pte_hbm[ti], toff}] <= wdata;
              pte_dirty[ti] <= 1'b1;
              respond(OK, '0);
            end else st <= READ2;
          end
          PARK, RESTORE, FREE: st <= SCAN;
          SHARE: begin
            if (npages[sess] != 0 || n > npages[src] || sess == src) respond(BADSTATE, '0);
            else st <= SCAN;
          end
          default: respond(BADSTATE, '0);
        endcase

        READ2: respond(OK, hbm_mem[{pte_hbm[ti], toff}]);

        // walk the session's pages; PARK/RESTORE hand single pages to COPY
        SCAN: begin
          if (op == SHARE) begin
            if (p == n) begin npages[sess] <= n; respond(OK, '0); end
            else if (!pte_in_hbm[si]) respond(BADSTATE, '0);      // source must be resident
            else begin
              pte_hbm[pi]    <= pte_hbm[si];
              pte_in_hbm[pi] <= 1'b1;
              pte_in_ddr[pi] <= 1'b0;
              pte_dirty[pi]  <= 1'b0;
              pte_shared[pi] <= 1'b1;
              pte_shared[si] <= 1'b1;                              // pin the source page too
              refcnt[pte_hbm[si]] <= refcnt[pte_hbm[si]] + 1'b1;
              p <= p + 1'b1;
            end
          end else if (p == npages[sess]) begin
            if (op == FREE) npages[sess] <= '0;
            respond(OK, DATA_W'(moved));
          end else if (op == FREE) begin
            if (pte_in_hbm[pi]) drop_hbm(pte_hbm[pi]);
            if (pte_in_ddr[pi]) ddr_free[pte_ddr[pi]] <= 1'b1;
            pte_in_hbm[pi] <= 1'b0; pte_in_ddr[pi] <= 1'b0; pte_dirty[pi] <= 1'b0; pte_shared[pi] <= 1'b0;
            p <= p + 1'b1;
          end else if (op == PARK) begin
            if (pte_shared[pi] || !pte_in_hbm[pi]) p <= p + 1'b1;          // pinned or already parked
            else if (!pte_dirty[pi] && pte_in_ddr[pi]) begin                // clean: DDR copy is current
              drop_hbm(pte_hbm[pi]);
              pte_in_hbm[pi] <= 1'b0;
              p <= p + 1'b1;
            end else if (!pte_in_ddr[pi] && !ddr_any) respond(NOSPACE, DATA_W'(moved));
            else begin                                                      // copy HBM -> DDR
              if (!pte_in_ddr[pi]) begin
                pte_ddr[pi] <= ddr_pick;
                ddr_free[ddr_pick] <= 1'b0;
                pte_in_ddr[pi] <= 1'b1;
              end
              copy_to_ddr <= 1'b1; w <= '0; st <= COPY;
            end
          end else begin                                                    // RESTORE
            if (pte_in_hbm[pi]) p <= p + 1'b1;
            else if (!pte_in_ddr[pi]) respond(BADSTATE, DATA_W'(moved));
            else if (!hbm_any) respond(NOSPACE, DATA_W'(moved));
            else begin                                                      // copy DDR -> HBM
              pte_hbm[pi] <= hbm_pick;
              hbm_free[hbm_pick] <= 1'b0;
              refcnt[hbm_pick] <= RC'(1);
              copy_to_ddr <= 1'b0; w <= '0; st <= COPY;
            end
          end
        end

        // one token per cycle; the page table entry was set up by SCAN
        COPY: begin
          if (copy_to_ddr) ddr_mem[{pte_ddr[pi], w}] <= hbm_mem[{pte_hbm[pi], w}];
          else             hbm_mem[{pte_hbm[pi], w}] <= ddr_mem[{pte_ddr[pi], w}];
          moved <= moved + 1;
          w <= w + 1'b1;
          if (w == OW'(PAGE_TOK - 1)) begin
            if (copy_to_ddr) begin
              stat_tok_parked <= stat_tok_parked + PAGE_TOK;
              pte_dirty[pi]   <= 1'b0;
              pte_in_hbm[pi]  <= 1'b0;
              drop_hbm(pte_hbm[pi]);
            end else begin
              stat_tok_restored <= stat_tok_restored + PAGE_TOK;
              pte_in_hbm[pi] <= 1'b1;
              pte_dirty[pi]  <= 1'b0;
            end
            p  <= p + 1'b1;
            st <= SCAN;
          end
        end

        RESP: if (rsp_ready) begin rsp_valid <= 1'b0; st <= IDLE; end
        default: st <= IDLE;
      endcase
    end
  end

  // free-page counters for the host and the testbench
  always_comb begin
    stat_hbm_free = '0;
    for (int i = 0; i < H; i++) stat_hbm_free += hbm_free[i];
    stat_ddr_free = '0;
    for (int i = 0; i < D; i++) stat_ddr_free += ddr_free[i];
  end
endmodule
