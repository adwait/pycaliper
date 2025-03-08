
module miter (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire stop,
    output wire [2:0] counter
    // output wire [1:0] state

`ifndef VERILOG
    // dummy
    , input wire [31:0] d
    , output wire [31:0] q
`endif
);

    fsm_counter a (
        .clk(clk),
        .rst(rst),
        .start(start),
        .stop(stop),
        .counter(counter)
`ifndef VERILOG
        , .d(d),
        .q(q)
`endif
    );

endmodule
