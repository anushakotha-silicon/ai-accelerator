#!/usr/bin/env python3
"""Create the cl_ia1 custom-logic project inside an aws-fpga (f2 branch) checkout.

Run on the FPGA Developer AMI after `source hdk_setup.sh`:

    python3 fpga/aws_f2/setup_cl.py --aws-fpga ~/aws-fpga

Steps:
  1. create_new_cl.py --new_cl_name cl_ia1   (AWS's own CL_TEMPLATE copy)
  2. copy rtl/*.sv and ocl_hookup.inc into cl_ia1/design
  3. in the CL top: comment out the template's OCL (cl_ocl_*) tie-offs and
     include ocl_hookup.inc before endmodule

No synthesis file list to edit: the shell's encrypt.tcl copies every
*.{v,sv,vh,svh,inc} in design/ and synth_cl_ia1.tcl reads every .sv/.v it copied.
The CL-top edit is verified; if the template text differs from what this
script expects, it stops and prints the manual edit instead of guessing.
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

CL = "cl_ia1"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RTL = ["pe.sv", "systolic_array.sv", "tile_core.sv", "kv_manager.sv", "kv_axil.sv", "axil_split2.sv", "ia1_top.sv"]

MANUAL = f"""
Manual edit: in hdk/cl/examples/{CL}/design/{CL}.sv, delete or comment out every
assignment that drives a cl_ocl_* output (the OCL tie-off block), then add
    `include "ocl_hookup.inc"
just before `endmodule`.
"""


def die(msg):
    print(f"ERROR: {msg}\n{MANUAL}", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aws-fpga", required=True, help="path to the aws-fpga checkout (f2 branch)")
    a = ap.parse_args()
    examples = os.path.join(os.path.abspath(os.path.expanduser(a.aws_fpga)), "hdk", "cl", "examples")
    cl_dir = os.path.join(examples, CL)

    # 1. new CL from AWS's template
    if not os.path.isdir(cl_dir):
        subprocess.run([sys.executable, "create_new_cl.py", "--new_cl_name", CL], cwd=examples, check=True)
    if not os.path.isdir(cl_dir):
        die(f"create_new_cl.py did not create {cl_dir}")
    design = os.path.join(cl_dir, "design")

    # 2. sources
    for f in RTL:
        shutil.copy(os.path.join(REPO, "rtl", f), design)
    shutil.copy(os.path.join(HERE, "ocl_hookup.inc"), design)

    # 3. CL top: replace OCL tie-offs with the core
    top = os.path.join(design, f"{CL}.sv")
    if not os.path.isfile(top):
        cands = glob.glob(os.path.join(design, "*.sv"))
        die(f"expected {top}; found {cands}")
    src = open(top).read()
    if "ocl_hookup.inc" not in src:
        tie = re.compile(r"^(\s*)(assign\s+cl_ocl_\w+.*?;|cl_ocl_\w+\s*(?:<=|=).*?;)\s*$", re.M)
        n = len(tie.findall(src))
        if n == 0:
            die("no cl_ocl_* tie-offs found in the CL top")
        src = tie.sub(lambda m: f"{m.group(1)}// IA-1: driven by ia1_top  {m.group(2)}", src)
        idx = src.rfind("endmodule")
        if idx < 0:
            die("no endmodule in the CL top")
        src = src[:idx] + '`include "ocl_hookup.inc"\n\n' + src[idx:]
        open(top, "w").write(src)
        print(f"patched {top}: commented {n} OCL tie-off statements, included ocl_hookup.inc")

    print(f"\n{CL} is ready. Build with:\n"
          f"  export CL_DIR={cl_dir}\n  cd $CL_DIR/build/scripts\n  ./aws_build_dcp_from_cl.py -c {CL}")


if __name__ == "__main__":
    main()
