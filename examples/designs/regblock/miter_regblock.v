module miter (
    clk,
    rst1,
    rst2,
    rd_index1,
    rd_index2,
    wr_index1,
    wr_index2,
    en1,
    en2,
    d1,
    d2,
    q1,
    q2
);

    input wire clk;
    input wire rst1;
    input wire rst2;
    input wire rd_index1;
    input wire rd_index2;
    input wire wr_index1;
    input wire wr_index2;
    input wire en1;
    input wire en2;
    input wire [31:0] d1;
    input wire [31:0] d2;
    output wire [31:0] q1;
    output wire [31:0] q2;

    regblock a (
        .clk(clk),
        .rst(rst1),
        .rd_index(rd_index1),
        .wr_index(wr_index1),
        .en(en1),
        .d(d1),
        .q(q1)
    );

    regblock b (
        .clk(clk),
        .rst(rst2),
        .rd_index(rd_index2),
        .wr_index(wr_index2),
        .en(en2),
        .d(d2),
        .q(q2)
    );


endmodule
