module adder(input clock, input reset);

    reg [32-1:0] ra;
    reg [32-1:0] rb;
    reg [32-1:0] ry;

    always @(posedge clock) begin
        if (reset) begin
            ra <= 0;
            rb <= 0;
            ry <= 0;
        end
        else begin
            ry <= ra + rb;
        end
    end

    initial begin
        $display("Running verilog adder...");
    end

endmodule
