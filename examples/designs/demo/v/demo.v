module fsm_counter (
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
	output reg [2:0] counter;
	reg [1:0] state;
	always @(posedge clk)
		if (!rst) begin
			state <= 2'd0;
			counter <= 3'd0;
		end
		else
			case (state)
				2'd0: begin
					counter <= 3'd0;
					if (start)
						state <= 2'd1;
				end
				2'd1: begin
					counter <= 3'd0;
					state <= 2'd2;
				end
				2'd2: begin
					counter <= counter + 1'b1;
					if ((stop && (counter >= 3'd5)) && (counter <= 3'd6))
						state <= 2'd3;
				end
				2'd3: state <= 2'd3;
				default: state <= 2'd0;
			endcase
endmodule
