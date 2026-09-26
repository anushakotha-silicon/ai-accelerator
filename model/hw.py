"""Hardware description: chip configuration, numeric formats and energy table.

All energy numbers are *assumptions* for a 5nm-class logic node with HBM3E,
scaled from published figures (Horowitz ISSCC'14, HBM/LPDDR vendor pJ/bit
claims). They are deliberately centralised here so they can be swapped for
real characterisation data later. See docs/ARCHITECTURE.md section 8.
"""
from dataclasses import dataclass, field

# Bytes per element. FP4 uses the OCP MX format: 4-bit elements + one 8-bit
# shared scale per 32 elements = 4.25 bits.
DTYPE_BYTES = {"fp4": 4.25 / 8, "fp8": 1.0, "bf16": 2.0}

# MACs per processing element per cycle, relative to FP8.
DTYPE_RATE = {"fp4": 2.0, "fp8": 1.0, "bf16": 0.5}


@dataclass(frozen=True)
class EnergyTable:
    """Picojoules per operation / per byte moved."""
    # Effective energy per PE-MAC: multiplier + accumulator + pipeline
    # registers + local clocking. The bare multiplier is ~3x cheaper; using
    # it alone would imply ~10 TFLOPS/W at chip level, which no shipping
    # part approaches.
    mac: dict = field(default_factory=lambda: {"fp4": 0.14, "fp8": 0.25, "bf16": 0.60})
    vec_mac_pj: float = 0.40          # attention engine MAC (less operand reuse than array)
    vec_op_pj: float = 1.00           # FP32 elementwise op (exp, norm, rope...)
    sram_pj_per_byte: float = 0.8     # tile scratchpad access
    noc_pj_per_byte_hop: float = 0.3
    hbm_pj_per_byte: float = 30.0     # ~3.75 pJ/bit incl. PHY + controller
    lpddr_pj_per_byte: float = 40.0   # ~5 pJ/bit
    c2c_pj_per_byte: float = 40.0     # chip-to-chip SerDes, ~5 pJ/bit


@dataclass(frozen=True)
class Node:
    """Process-node scaling relative to the 5nm-class baseline.

    Logic scales well from N5 to N3E; SRAM bitcells, wires and analog PHYs
    barely scale (the N3E high-density bitcell is 0.021 um^2, same as N5).
    That asymmetry decides which blocks get cheaper.
    """
    name: str
    logic_energy: float      # dynamic energy per logic op at nominal V (N5 = 1)
    sram_energy: float       # energy per SRAM byte
    wire_energy: float       # energy per byte per NoC hop
    logic_area: float        # area per logic function
    sram_area: float         # area per SRAM bit
    tile_static_w: float     # leakage + clock tree per tile at nominal V/f
    fixed_static_w: float    # PHY idle, control cores, I/O: not on the compute DVFS rail


N5 = Node("5nm-class", 1.0, 1.0, 1.0, 1.0, 1.0, 1.25, 20.0)
# TSMC's public N3E-vs-N5 claim: ~32% lower power at iso-speed, ~1.6x logic density.
N3 = Node("3nm-class (N3E)", 0.68, 0.85, 0.85, 0.625, 1.0, 1.0, 17.0)

# Operating points for the compute rail (tiles + control spine): (volts, GHz).
V_NOM, F_NOM = 0.75, 1.2
OPPS = ((0.60, 0.80), (0.675, 1.00), (V_NOM, F_NOM), (0.85, 1.45))

# Die-area model (mm^2 at N5, 32-tile floorplan in docs/package-3d.html).
AREA = {"array": 2.89, "sram_per_mb": 1.75 / 4, "vector": 0.66, "router": 0.30, "tile_ws": 1.69,
        "hbm_phy": 18.7, "lpddr_phy": 22.44, "io": 22.1, "spine": 46.64, "die_ws": 42.7}


@dataclass(frozen=True)
class Chip:
    name: str = "baseline"
    # --- compute tiles ---
    tiles: int = 32
    array_dim: int = 128                  # array_dim x array_dim FP8 MAC systolic array per tile
    clock_ghz: float = 1.2
    weight_load_bytes_per_cycle: int = 256  # per tile: 2 array rows/cycle at FP8
    vec_macs_per_cycle: int = 1024        # per tile, attention / vector engine (FP8/BF16 dot)
    vec_ops_per_cycle: int = 512          # per tile, FP32 elementwise incl. exp
    sram_mb_per_tile: float = 4.0
    # --- memory ---
    hbm_gb: float = 144.0                 # 4 x 36 GB HBM3E stacks
    hbm_tbps: float = 4.8
    lpddr_gb: float = 256.0               # capacity tier for parked KV; 0 disables
    lpddr_tbps: float = 0.5
    # --- efficiencies & overheads ---
    mem_eff: float = 0.85                 # achievable fraction of DRAM peak
    mm_eff: float = 0.85                  # achievable fraction of array peak
    layer_sync_us: float = 0.5            # per-layer barrier / sequencer cost
    tdp_w: float = 350.0
    energy: EnergyTable = field(default_factory=EnergyTable)
    # --- process & power management ---
    node: Node = N5
    vdd: float = V_NOM                    # compute-rail voltage at the current operating point
    dvfs: str = "off"                     # "off" | "efficiency" | "performance"
    power_gating: bool = False            # gate idle arrays (keeps SRAM retention)
    speed_slack: float = 0.02             # scheduler may be this much slower than the fastest option to save energy
    hbm_stacks: int = 4

    # derived ---------------------------------------------------------------
    @property
    def clock_hz(self) -> float:
        return self.clock_ghz * 1e9

    @property
    def pes(self) -> int:
        return self.tiles * self.array_dim ** 2

    def peak_macs(self, dtype: str) -> float:
        return self.pes * DTYPE_RATE[dtype] * self.clock_hz

    def peak_tflops(self, dtype: str) -> float:
        return 2 * self.peak_macs(dtype) / 1e12

    @property
    def hbm_bw(self) -> float:
        return self.hbm_tbps * 1e12 * self.mem_eff

    @property
    def lpddr_bw(self) -> float:
        return self.lpddr_tbps * 1e12 * self.mem_eff

    @property
    def sram_mb(self) -> float:
        return self.tiles * self.sram_mb_per_tile

    @property
    def v_scale(self) -> float:
        """Dynamic energy scales with V^2."""
        return (self.vdd / V_NOM) ** 2

    def tile_static_w_at(self, vdd: float, ghz: float) -> float:
        # half leakage (~V^3 over this range), half clock tree (C V^2 f)
        r = vdd / V_NOM
        return self.tiles * self.node.tile_static_w * (0.5 * r ** 3 + 0.5 * r ** 2 * ghz / F_NOM)

    @property
    def static_w(self) -> float:
        """Static power at nominal voltage and frequency."""
        return self.node.fixed_static_w + self.tile_static_w_at(V_NOM, F_NOM)

    @property
    def idle_static_w(self) -> float:
        """Power while the chip waits for work (e.g. every agent is in a tool call)."""
        if self.dvfs == "off":
            return self.static_w
        v, f = OPPS[0]
        tiles = self.tile_static_w_at(v, f)
        if self.power_gating:
            tiles *= 1 - 0.5 * 0.8        # arrays are half of tile static; gating removes 80%
        return self.node.fixed_static_w + tiles

    @property
    def die_area_mm2(self) -> float:
        n, a = self.node, AREA
        tile = ((a["array"] + a["vector"] + a["router"]) * n.logic_area
                + a["sram_per_mb"] * self.sram_mb_per_tile * n.sram_area
                + a["tile_ws"] * (0.8 if n.logic_area < 1 else 1.0))
        return (self.tiles * tile + a["hbm_phy"] * self.hbm_stacks
                + (a["lpddr_phy"] if self.lpddr_gb > 0 else 0) + a["io"]
                + a["spine"] * n.logic_area + a["die_ws"] * (0.8 if n.logic_area < 1 else 1.0))

    @property
    def noc_hops(self) -> float:
        # 2D mesh with memory controllers on the edges: mean ~ sqrt(tiles)/2
        return max(1.0, self.tiles ** 0.5 / 2)


@dataclass(frozen=True)
class Serving:
    """Software policy choices the hardware must support."""
    weight_dtype: str = "fp4"
    compute_dtype: str = "fp8"            # activations; FP4 weights dequantised in-array
    kv_dtype: str = "fp8"
    prefill_chunk: int = 4096
    spec_k: int = 0                        # draft tokens per step (prompt-lookup drafting)
    spec_accept: float = 0.0               # per-token acceptance probability

    def tokens_per_step(self) -> float:
        """Expected tokens emitted per sequence per verify step."""
        if self.spec_k == 0:
            return 1.0
        a = self.spec_accept
        if a >= 1.0:
            return self.spec_k + 1.0
        return (1 - a ** (self.spec_k + 1)) / (1 - a)
