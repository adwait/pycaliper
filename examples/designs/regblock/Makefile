
V_FILES=miter.v regblock.v

btor/%.btor: %.v
	yosys -p "read_verilog $<; proc; opt; memory -nomap; opt; dffunmap; clk2fflogic; write_btor $@"

btor: $(V_FILES:%.v=btor/%.btor)

btor/reg_en.btor: miter_reg_en.v regblock.v
	yosys -p "read_verilog $^; hierarchy -top miter; hierarchy -check; proc; opt; memory; flatten; clk2fflogic; write_btor btor/reg_en.btor"

btor/regblock.btor: miter_regblock.v regblock.v
	yosys -p "read_verilog $^; hierarchy -top miter; hierarchy -check; proc; opt; memory; flatten; clk2fflogic; write_btor btor/regblock.btor"

clean:
	rm -f v/*.v btor/*.btor

.PHONY: clean
