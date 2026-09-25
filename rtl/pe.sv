// Processing element for the weight-stationary systolic array.
//
// Holds two weight registers (double buffer). Each activation arrives with a
// one-bit tag naming the buffer it must multiply against, so a block switch
// travels through the array with the data itself: no global "swap" signal.
//
//   act  ->  flows left to right, one PE per cycle
//   psum ->  flows top to bottom, accumulating act * w
//
// M1 datapath: INT8 x INT8 -> INT32 accumulate (exact, easy to check).
// M2 replaces the multiplier with FP8/MXFP4 and FP32 accumulation.
module pe #(
  parameter int AW   = 8,
  parameter int ACCW = 32
) (
  input  logic            clk,
  input  logic            rst_n,
  // activation from the left, tagged with the weight buffer it uses
  input  logic            act_valid_i,
  input  logic            act_buf_i,
  input  logic [AW-1:0]   act_i,
  // partial sum from above
  input  logic [ACCW-1:0] psum_i,
  // weight write, already skewed to this column by the array
  input  logic            w_we,
  input  logic            w_buf,
  input  logic [AW-1:0]   w_data,
  // to the right / below
  output logic            act_valid_o,
  output logic            act_buf_o,
  output logic [AW-1:0]   act_o,
  output logic [ACCW-1:0] psum_o
);
  logic [AW-1:0]          w0, w1;
  logic signed [2*AW-1:0] prod;
  logic [ACCW-1:0]        prod_ext;

  assign prod     = $signed(act_i) * $signed(act_buf_i ? w1 : w0);
  assign prod_ext = act_valid_i ? {{(ACCW - 2 * AW){prod[2*AW-1]}}, prod} : '0;

  // A write and a read of the same buffer on the same edge is safe: the
  // multiply above sees the old value. The sequencer guarantees the last use
  // of a buffer happens no later than the write edge.
  always_ff @(posedge clk) begin
    if (w_we && !w_buf) w0 <= w_data;
    if (w_we &&  w_buf) w1 <= w_data;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      act_valid_o <= 1'b0;
      act_buf_o   <= 1'b0;
      act_o       <= '0;
      psum_o      <= '0;
    end else begin
      act_valid_o <= act_valid_i;
      act_buf_o   <= act_buf_i;
      act_o       <= act_i;
      psum_o      <= psum_i + prod_ext;
    end
  end
endmodule
