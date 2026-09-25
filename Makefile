# Phase 2 RTL flow. Needs Icarus Verilog (iverilog, vvp); lint needs Verilator.
N      ?= 8
LANES  ?= 2
M      ?= 4
BLOCKS ?= 8
SEED   ?= 1
EARLY  ?= 0
BUILD  := build
RTL    := rtl/pe.sv rtl/systolic_array.sv

.PHONY: sim sweep hazard lint clean

# one run: generate vectors, compile, simulate, print RESULT line
sim:
	@mkdir -p $(BUILD)
	@python3 tb/gen_vectors.py --n $(N) --lanes $(LANES) --m $(M) --blocks $(BLOCKS) --seed $(SEED) --out $(BUILD)
	@iverilog -g2012 -I $(BUILD) -o $(BUILD)/sim $(RTL) tb/tb_systolic_array.sv
	@vvp -n $(BUILD)/sim +EARLY=$(EARLY)

# cycles per block vs the model's max(M, N/LANES), across block sizes
sweep:
	@for m in 1 2 3 4 5 8 16 32; do $(MAKE) --no-print-directory sim M=$$m SEED=$$m | grep RESULT; done

# start a weight load one row too early: must fail, proving the window is exact
hazard:
	@$(MAKE) --no-print-directory sim M=8 EARLY=1 | grep -E "RESULT|MISMATCH"

lint:
	verilator --lint-only -Wall -Wno-DECLFILENAME --top-module systolic_array $(RTL)

clean:
	rm -rf $(BUILD)
