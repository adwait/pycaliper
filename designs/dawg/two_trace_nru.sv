// Parent module with a miter with different inputs
module miter (
    input wire clk
    , input wire rst
    , output wire [31:0] qA
    , output wire [31:0] qB
);


    cacheline a (
        .clk(clk)
        , .reset(rst)
    );

    cacheline b (
        .clk(clk)
        , .reset(rst)
    );

    default clocking cb @(posedge clk);
    endclocking // cb

    logic attacker_domain;
    logic [7:0] attacker_hitmap;

    always_ff @(posedge clk) begin
        attacker_domain <= attacker_domain;
        attacker_hitmap <= attacker_hitmap;
    end


    logic fvreset;

    `include "cacheline_nru.pyc.sv"

endmodule
