// Parent module with a miter with different inputs
module miter (
    input wire clk
    , input wire rst
);


    sdram_controller a (
        .clk(clk)
        , .rst_n(rst)
    );

    // sdram_controller b (
    //     .clk(clk)
    //     , .rst_n(rst)
    // );

    default clocking cb @(posedge clk);
    endclocking // cb

    logic fvreset;

    `include "sdram.pyc.sv"

endmodule
