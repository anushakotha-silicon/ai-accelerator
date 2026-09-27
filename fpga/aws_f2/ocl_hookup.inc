// IA-1 tile core on the AWS F2 OCL (AppPF BAR0, AXI4-Lite) port.
// Included inside the CL module by setup_cl.py (.inc: the extension the
// shell's encrypt.tcl copies and encrypts with the design sources), in place of the template's
// OCL tie-offs. Signal names follow hdk/common/shell_stable/design/interfaces/cl_ports.vh.
//
// N = 32 x 32 INT8 array (1,024 multipliers; VU47P has 9,024 DSP slices, so
// N = 64 also fits). Clock: clk_main_a0 from the shell's clock recipe.
// ocl_cl_wstrb is unused: the core only accepts full 32-bit writes.

tile_core #(
  .N          (32),
  .LANES      (2),
  .MAX_BLOCKS (16),
  .MAX_M      (64)
) u_ia1_tile (
  .clk       (clk_main_a0),
  .rst_n     (rst_main_n),
  .s_awaddr  (ocl_cl_awaddr),
  .s_awvalid (ocl_cl_awvalid),
  .s_awready (cl_ocl_awready),
  .s_wdata   (ocl_cl_wdata),
  .s_wvalid  (ocl_cl_wvalid),
  .s_wready  (cl_ocl_wready),
  .s_bresp   (cl_ocl_bresp),
  .s_bvalid  (cl_ocl_bvalid),
  .s_bready  (ocl_cl_bready),
  .s_araddr  (ocl_cl_araddr),
  .s_arvalid (ocl_cl_arvalid),
  .s_arready (cl_ocl_arready),
  .s_rdata   (cl_ocl_rdata),
  .s_rresp   (cl_ocl_rresp),
  .s_rvalid  (cl_ocl_rvalid),
  .s_rready  (ocl_cl_rready)
);
