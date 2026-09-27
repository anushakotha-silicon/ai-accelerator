// End-to-end test of tile_core through its AXI4-Lite port, exactly as the
// AWS F2 host will drive it: load buffers word by word, start, poll, read back.
`timescale 1ns/1ps
`ifndef VEC_DIR
  `define VEC_DIR "build"
`endif
module tb_tile_core;
  `include "params.svh"
  localparam int AW = 8, ACCW = 32, MAX_BLOCKS = 16, MAX_M = 64;
  localparam int WPR = N * AW / 32;

  logic clk = 1'b0, rst_n = 1'b0;
  always #2 clk = ~clk;                       // 250 MHz, the F2 target clock

  logic [31:0] awaddr, wdata, araddr, rdata;
  logic awvalid, awready, wvalid, wready, bvalid, bready, arvalid, arready, rvalid, rready;
  logic [1:0] bresp, rresp;

  tile_core #(.N(N), .LANES(LANES), .MAX_BLOCKS(MAX_BLOCKS), .MAX_M(MAX_M)) dut (
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

  // --------------------------------------------------------- AXI-Lite BFM
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

  // --------------------------------------------------------------- test
  initial begin
    logic [31:0] v, cycles, stalls, s1, sl, id;
    integer errors;
    real period;
    integer model;
    errors = 0;
    awvalid = 0; wvalid = 0; bready = 0; arvalid = 0; rready = 0;
    awaddr = 0; wdata = 0; araddr = 0;
    $readmemh({`VEC_DIR, "/weights.hex"}, wmem);
    $readmemh({`VEC_DIR, "/acts.hex"}, xmem);
    $readmemh({`VEC_DIR, "/expected.hex"}, ymem);
    repeat (4) @(posedge clk);
    rst_n = 1;

    axi_read(32'h0, id);
    if (id !== 32'h1A1C_0001) begin $display("BAD ID %h", id); errors++; end
    axi_write(32'hC, NB);
    axi_write(32'h10, M);

    // weights: row (b*N + k) holds W[b][k][0..N-1], 4 elements per word
    for (int r = 0; r < NB * N; r++)
      for (int w = 0; w < WPR; w++)
        axi_write(32'h10_0000 + (r * WPR + w) * 4,
                  {wmem[r*N + 4*w + 3], wmem[r*N + 4*w + 2], wmem[r*N + 4*w + 1], wmem[r*N + 4*w]});
    // activations: row (b*M + m) holds X[b][m][0..N-1]
    for (int r = 0; r < NB * M; r++)
      for (int w = 0; w < WPR; w++)
        axi_write(32'h20_0000 + (r * WPR + w) * 4,
                  {xmem[r*N + 4*w + 3], xmem[r*N + 4*w + 2], xmem[r*N + 4*w + 1], xmem[r*N + 4*w]});

    axi_write(32'h4, 32'h1);                    // start
    do axi_read(32'h8, v); while (!v[1]);       // poll done

    for (int r = 0; r < NB * M; r++)
      for (int n = 0; n < N; n++) begin
        axi_read(32'h30_0000 + (r * N + n) * 4, v);
        if (v !== ymem[r*N + n]) begin
          errors++;
          if (errors <= 5) $display("MISMATCH row %0d col %0d: got %0d expected %0d", r, n, $signed(v), $signed(ymem[r*N + n]));
        end
      end

    axi_read(32'h14, cycles);
    axi_read(32'h18, stalls);
    axi_read(32'h1C, s1);
    axi_read(32'h20, sl);
    period = $itor(sl - s1) / (NB - 2);
    model  = (M > (N + LANES - 1) / LANES) ? M : (N + LANES - 1) / LANES;
    $display("RESULT tile_core N=%0d LANES=%0d M=%0d blocks=%0d | cycles=%0d stalls=%0d | cycles/block measured=%0.2f model=%0d | errors=%0d | %s",
             N, LANES, M, NB, cycles, stalls, period, model, errors, errors == 0 ? "PASS" : "FAIL");
    $finish;
  end

  initial begin #(50_000_000); $display("RESULT TIMEOUT"); $finish; end
endmodule
