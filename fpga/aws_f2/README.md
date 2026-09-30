# IA-1 tile on AWS F2

Runs the IA-1 FPGA top (`rtl/ia1_top.sv`) on an AWS EC2 F2 FPGA: the tile core
(32×32 INT8 systolic array, wavefront weight loading, hardware sequencer) and the
agent KV manager (paging, park/restore, incremental parking, prefix sharing),
both behind the shell's OCL AXI-Lite port, checked against the same golden
models and scoreboards the simulations use.

```
host (F2 instance) ── PCIe BAR0 ── AWS Small Shell ── OCL AXI-Lite ── ia1_top ─┬─ tile_core  (0x000000)
  ia1_host.py, kv_host.py                             ocl_hookup.inc         └─ kv_manager (0x400000)
```

Everything here was verified locally before any AWS spend: `make top` runs a
matmul and a multi-agent KV scenario through the single AXI-Lite port (1,966
checks, 0 errors); `make lint-top` is clean; the hookup lints against the shell's
port names; `setup_cl.py` was tested on a mock of AWS's template; and both host
programs pass end to end with `--fake` (a software stand-in for BAR0).
**Not yet verified:** Vivado synthesis, timing closure and real hardware. Those need AWS.

## What you need

| | |
|---|---|
| AWS account | with F2 access (F2 instance quotas may need a support request) |
| Build machine | the free **FPGA Developer AMI** (Vivado preinstalled). AWS recommends ≥4 vCPU and 32 GiB for F2 builds. |
| Run machine | an **F2** instance (the smallest size has one FPGA) |
| S3 bucket | to hand the build to AWS's AFI (Amazon FPGA Image) service |

Check current EC2 pricing before starting: builds take hours, and F2 instances bill by the hour. Stop them when idle.

## 1. Build (on the FPGA Developer AMI)

```bash
git clone -b f2 https://github.com/aws/aws-fpga.git ~/aws-fpga
cd ~/aws-fpga && source hdk_setup.sh
git clone https://github.com/anushakotha-silicon/ai-accelerator.git ~/ai-accelerator
python3 ~/ai-accelerator/fpga/aws_f2/setup_cl.py --aws-fpga ~/aws-fpga
export CL_DIR=~/aws-fpga/hdk/cl/examples/cl_ia1
cd $CL_DIR/build/scripts && ./aws_build_dcp_from_cl.py -c cl_ia1
```

`setup_cl.py` creates `cl_ia1` from AWS's own `CL_TEMPLATE`, copies our RTL in, and
replaces only the template's OCL tie-offs with `ocl_hookup.inc`. Every other shell
tie-off stays as AWS wrote it. No synthesis file list needs editing: the shell's
`encrypt.tcl` copies everything in `design/`.

After the build, check the timing summary in `$CL_DIR/build/reports` before
spending on an AFI. Negative slack means lowering the clock or pipelining (see below).

## 2. Create the AFI

Upload the build tarball (the `.tar` the build reports under `$CL_DIR/build/checkpoints`) and register it:

```bash
aws s3 cp <build>.tar s3://<bucket>/ia1/
aws ec2 create-fpga-image --name cl_ia1 \
  --input-storage-location Bucket=<bucket>,Key=ia1/<build>.tar \
  --logs-storage-location Bucket=<bucket>,Key=ia1/logs/
aws ec2 describe-fpga-images --fpga-image-ids <afi-id>     # wait for "available"
```

AWS also ships `hdk/scripts/create_afi.py`, which walks through the same steps.

## 3. Run (on the F2 instance)

```bash
git clone -b f2 https://github.com/aws/aws-fpga.git ~/aws-fpga && cd ~/aws-fpga && source sdk_setup.sh
sudo fpga-load-local-image -S 0 -I <agfi-id>
git clone https://github.com/anushakotha-silicon/ai-accelerator.git ~/ai-accelerator
sudo python3 ~/ai-accelerator/fpga/aws_f2/host/ia1_host.py --slot 0 --m 4 --blocks 12
sudo python3 ~/ai-accelerator/fpga/aws_f2/host/ia1_host.py --slot 0 --m 1 --blocks 16   # decode-like
sudo python3 ~/ai-accelerator/fpga/aws_f2/host/kv_host.py --slot 0 --agents 4 --turns 3   # agent memory
```

Try the host programs locally first, without an FPGA:

```bash
python3 fpga/aws_f2/host/ia1_host.py --fake --m 4 --blocks 12
python3 fpga/aws_f2/host/kv_host.py --fake --agents 4 --turns 3
```

Expected, matching simulation:

```
RESULT fpga N=32 LANES=2 M=4 blocks=12 | cycles=... | cycles/block measured=16.00 model=16 | errors=0 | PASS
park    : 384 tokens, ~484 cycles (~1.26 cycles/token), host wall time ... us/token incl. PCIe
RESULT fpga-kv agents=4 turns=3 | checks=1877 errors=0 | PASS
```

KV cycle counts come from the hardware (`KV_CYCLES`): about one cycle per token
copied plus page-table overhead. Host wall time is dominated by PCIe round trips,
since every register access crosses the bus.

## What the FPGA run proves, and what it doesn't

| Proves | Doesn't prove |
|---|---|
| The wavefront loader and sequencer work on real silicon at real clock rates | ASIC energy: the same logic on an FPGA draws roughly 12× the dynamic power (Kuon & Rose's FPGA-vs-ASIC study), so energy still comes from the model until an ASIC test chip |
| Cycles per block = max(M, N/LANES) measured by hardware counters | Throughput at scale: host loads buffers over AXI-Lite, one word at a time |
| The design closes timing (or shows where it doesn't) | The FP8/FP4 datapath (M2) and agent-memory blocks (next milestones) |

## Next steps on F2

1. **Bulk data path:** move buffer loads from OCL (one word per access) to PCIS
   (512-bit AXI4 bursts), so large runs aren't limited by the host.
2. **Card memory:** stream weights from the F2 card's HBM through the wavefront
   lanes, like the real chip streams from HBM.
3. **Real memory tiers for the KV manager:** today the two tiers are on-chip
   arrays. Backing them with the card's HBM (fast tier) and DDR (capacity tier)
   through AXI4 bursts measures park/restore bandwidth on real memory. The host
   interface stays the same.

If timing fails at the default clock: the array is local (neighbour-to-neighbour),
so the likely critical paths are the buffer reads (wide rows from BRAM/URAM) and
the AXI read mux. Both can take an extra pipeline register without changing the
schedule, since array inputs are already registered one stage after the sequencer.
