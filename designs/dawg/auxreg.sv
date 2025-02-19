
// Auxilliary register with variable width
module auxreg #(
        parameter WIDTH = 8
    ) (
        input logic clk,
        // input logic [WIDTH-1:0] din,
        output logic [WIDTH-1:0] dout
    );

    logic [WIDTH-1:0] auxreg;

    // always_ff @(posedge clk)
    //     auxreg <= din;

    assign dout = auxreg;
endmodule
