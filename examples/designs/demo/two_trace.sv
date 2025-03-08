// Parent module with a miter with different inputs
module miter (
    input wire clk
    , input wire rst
);

    fsm_counter a (
        .clk(clk)
        , .rst(rst)
    );

    fsm_counter b (
        .clk(clk)
        , .rst(rst)
    );

    default clocking cb @(posedge clk);
    endclocking // cb

    logic fvreset;

    `include "demo.pyc.sv"

endmodule
