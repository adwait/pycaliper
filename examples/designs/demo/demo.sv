module fsm_counter (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire stop,
    output logic [2:0] counter
    // output wire [1:0] state

`ifndef VERILOG
    // dummy
    , input wire [31:0] d
    , output wire [31:0] q
`endif
);

    // 2-bit state encoding using SystemVerilog typedef enum
    typedef enum logic [1:0] {
        IDLE = 2'd0,
        INIT = 2'd1,
        RUN  = 2'd2,
        DONE = 2'd3
    } state_t;

    // Internal signals for state and counter
    state_t state;
    // logic [2:0] counter;

    // Synchronous state update and counter logic
    always @(posedge clk) begin
        if (!rst) begin
            // Reset condition
            state   <= IDLE;
            counter <= 3'd0;
        end
        else begin
            case (state)
                IDLE: begin
                    // Remain in IDLE until 'start' is 1
                    counter <= 3'd0;  // keep counter at 0 in IDLE
                    if (start) begin
                        state <= INIT;
                    end
                end

                INIT: begin
                    // Initialize counter to 0, then go to RUN
                    counter <= 3'd0;
                    state   <= RUN;
                end

                RUN: begin
                    // Increment counter each cycle in RUN
                    counter <= counter + 1'b1;
                    // Transition to DONE if stop && (counter >= 5)
                    if (stop && (counter >= 3'd5) && (counter <= 3'd6)) begin
                        state <= DONE;
                    end
                end

                DONE: begin
                    // Once in DONE, remain here
                    state <= DONE;
                end

                default: begin
                    // Default for completeness
                    state <= IDLE;
                end
            endcase
        end
    end

`ifndef VERILOG
    reg_en regm (
        .clk(clk)
        , .rst(rst)
        , .en(start)
        , .d(d)
        , .q(q)
    );
`endif


endmodule

`ifndef VERILOG
module reg_en (
    input wire clk,
    input wire rst,
    input wire en,
    input wire [31:0] d,
    output wire [31:0] q
);

    logic [31:0] q;

    always @(posedge clk) begin
        if (rst) begin
            q <= 32'h0;
        end else if (en) begin
            q <= d;
        end
    end

endmodule
`endif
