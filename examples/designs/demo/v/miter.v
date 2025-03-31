module miter (
	clk,
	rst,
	start,
	stop,
	counter
);
	input wire clk;
	input wire rst;
	input wire start;
	input wire stop;
	output wire [2:0] counter;
	fsm_counter a(
		.clk(clk),
		.rst(rst),
		.start(start),
		.stop(stop),
		.counter(counter)
	);
endmodule
