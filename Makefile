# Phase 2 RTL flow. Needs Icarus Verilog (iverilog, vvp); lint needs Verilator.
# Tools built from source live in ~/.local/eda/bin when present; otherwise use PATH.
EDA       := $(HOME)/.local/eda/bin
tool       = $(if $(wildcard $(EDA)/$(1)),$(EDA)/$(1),$(1))
IVERILOG  := $(call tool,iverilog)
VVP       := $(call tool,vvp)
VERILATOR := $(call tool,verilator)
N      ?= 8
LANES  ?= 2
M      ?= 4
BLOCKS ?= 8
SEED   ?= 1
EARLY  ?= 0
# one directory per configuration, so parallel or background runs never share vectors
BUILD  := build/n$(N)_l$(LANES)_m$(M)_b$(BLOCKS)_s$(SEED)
RTL    := rtl/pe.sv rtl/systolic_array.sv
CORE   := $(RTL) rtl/tile_core.sv

.PHONY: sim sweep hazard regress core kv lint lint-core lint-kv clean

# one run: generate vectors, compile, simulate, print RESULT line
sim:
	@mkdir -p $(BUILD)
	@python3 tb/gen_vectors.py --n $(N) --lanes $(LANES) --m $(M) --blocks $(BLOCKS) --seed $(SEED) --out $(BUILD)
	@$(IVERILOG) -g2012 -I $(BUILD) -DVEC_DIR='"$(BUILD)"' -o $(BUILD)/sim $(RTL) tb/tb_systolic_array.sv
	@$(VVP) -n $(BUILD)/sim +EARLY=$(EARLY)

# cycles per block vs the model's max(M, N/LANES), across block sizes
sweep:
	@for m in 1 2 3 4 5 8 16 32; do $(MAKE) --no-print-directory sim M=$$m SEED=$$m | grep RESULT; done

# start a weight load one row too early: must fail, proving the window is exact
hazard:
	@$(MAKE) --no-print-directory sim M=8 EARLY=1 | grep -E "RESULT|MISMATCH"

# array sizes x lane counts x block sizes (both lane-bound and array-bound regimes)
regress:
	@for cfg in 8:1 8:2 16:2 16:4; do n=$${cfg%:*}; l=$${cfg#*:}; \
	  for m in 1 3 $$((n/l)) $$((2*n)); do \
	    $(MAKE) --no-print-directory sim N=$$n LANES=$$l M=$$m BLOCKS=16 SEED=$$((m+n)) | grep RESULT; done; done

# tile_core end to end through AXI-Lite (the FPGA-facing block)
core:
	@mkdir -p $(BUILD)
	@python3 tb/gen_vectors.py --n $(N) --lanes $(LANES) --m $(M) --blocks $(BLOCKS) --seed $(SEED) --out $(BUILD)
	@$(IVERILOG) -g2012 -I $(BUILD) -DVEC_DIR='"$(BUILD)"' -o $(BUILD)/core $(CORE) tb/tb_tile_core.sv
	@$(VVP) -n $(BUILD)/core

# agent KV manager: paging, park/restore, incremental parking, prefix sharing
kv:
	@mkdir -p build/kv
	@$(IVERILOG) -g2012 -o build/kv/sim rtl/kv_manager.sv tb/tb_kv_manager.sv
	@$(VVP) -n build/kv/sim

lint-kv:
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME --top-module kv_manager rtl/kv_manager.sv

lint-core:
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME --top-module tile_core $(CORE)

lint:
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME --top-module systolic_array $(RTL)

clean:
	rm -rf build
