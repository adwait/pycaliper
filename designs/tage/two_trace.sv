// Parent module with a miter with different inputs
module miter (
    input wire clk
    , input wire rst
);


    top a (
        .clk_i(clk)
        , .rst_i(rst)
    );

    top b (
        .clk_i(clk)
        , .rst_i(rst)
    );

    default clocking cb @(posedge clk);
    endclocking // cb

    logic fvreset;

    `include "tage.pyc.sv"

endmodule
