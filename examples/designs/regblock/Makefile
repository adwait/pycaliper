
V_FILES=miter.v regblock.v

btor/%.btor: %.v
	yosys -p "read_verilog $<; proc; opt; memory -nomap; opt; dffunmap; clk2fflogic; write_btor $@"

btor: $(V_FILES:%.v=btor/%.btor)

btor_full_design: $(V_FILES)
	yosys -p "read_verilog $^; hierarchy -top miter; hierarchy -check; proc; opt; memory; flatten; clk2fflogic; write_btor btor/full_design.btor"

clean:
	rm -f v/*.v btor/*.btor

.PHONY: clean
