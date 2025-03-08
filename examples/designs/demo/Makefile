
SV_FILES=miter.sv demo.sv

V_FILES=$(SV_FILES:%.sv=v/%.v)

v/%.v: %.sv
	sv2v -D VERILOG $< > $@

btor/%.btor: v/%.v
	yosys -p "read_verilog $<; proc; opt; memory -nomap; opt; dffunmap; clk2fflogic; write_btor $@"

verilog: $(V_FILES)

verilog_full_design: $(V_FILES)
	yosys -p "read_verilog $^; hierarchy -top top; hierarchy -check; proc; opt; memory; flatten; opt; write_verilog v/full_design.v"

btor: $(SV_FILES:%.sv=btor/%.btor)

btor_full_design: $(V_FILES)
	yosys -p "read_verilog $^; hierarchy -top miter; hierarchy -check; proc; opt; memory; flatten; clk2fflogic; write_btor btor/full_design.btor"

clean:
	rm -f v/*.v btor/*.btor

.PHONY: clean
