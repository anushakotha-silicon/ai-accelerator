"""PCIe BAR0 access to the IA-1 FPGA top on AWS F2 (shared by the host programs).

BAR0 is the shell's OCL AXI-Lite window: tile core at 0x000000, KV manager at
0x400000 (see rtl/ia1_top.sv). Needs root to mmap the device's resource0 file.
"""
import glob
import mmap
import os
import sys

AMAZON_VENDOR = 0x1D0F
BAR0_SIZE = 64 << 20

def find_bar0(slot):
    """Resource0 files of Amazon FPGA functions exposing a 64 MiB BAR0 (the AppPF)."""
    devs = []
    for d in sorted(glob.glob("/sys/bus/pci/devices/*")):
        try:
            vendor = int(open(os.path.join(d, "vendor")).read(), 16)
        except OSError:
            continue
        res0 = os.path.join(d, "resource0")
        if vendor == AMAZON_VENDOR and os.path.exists(res0) and os.path.getsize(res0) == BAR0_SIZE:
            devs.append(res0)
    if slot >= len(devs):
        sys.exit(f"no F2 AppPF BAR0 for slot {slot}; found {devs}. Is the AFI loaded?")
    return devs[slot]


class Bar:
    def __init__(self, path):
        self.fd = os.open(path, os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, BAR0_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        self.w = memoryview(self.mm).cast("I")          # 32-bit accesses only

    def peek(self, addr):
        return self.w[addr >> 2]

    def poke(self, addr, value):
        self.w[addr >> 2] = value & 0xFFFFFFFF
