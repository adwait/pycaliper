// module reg_en (
//     input wire clk,
//     input wire rst,
//     input wire en,
//     input wire [31:0] d,
//     output wire [31:0] q
// );

//     logic [31:0] q;

//     always @(posedge clk) begin
//         if (rst) begin
//             q <= 32'h0;
//         end else if (en) begin
//             q <= d;
//         end
//     end

// endmodule

module miter (
    clk,
    rst,
    en,
    d,
    q
);
    input wire clk;
    input wire rst;
    input wire en;
    input wire [31:0] d;
    output wire [31:0] q;

    reg_en a (
        .clk(clk),
        .rst(rst),
        .en(en),
        .d(d),
        .q(q)
    );
endmodule
