module driver (
    input  logic          clock,
    input  logic          reset,
    input  logic [32-1:0] opcode,
    input  logic [32-1:0] id,
    input  logic [32-1:0] in,
    input  logic [32-1:0] addr,
    output logic [32-1:0] out
);

    function void write_reg_a;
        input int value;
        input int addr;
        logic [31:0] tmp;
        begin
            tmp[0+:32] = 0;
            tmp[0+:32] = driver.dut.ra;
            tmp[addr*32+:32] = value;
            driver.dut.ra = tmp[0+:32];
        end
    endfunction

    function int read_reg_a;
        input int addr;
        logic [32-1:0] tmp;
        begin
            tmp[0+:32] = 0;
            tmp[0+:32] = driver.dut.ra;
            return tmp[addr*32+:32];
        end
    endfunction

    function void write_reg_b;
        input int value;
        input int addr;
        logic [31:0] tmp;
        begin
            tmp[0+:32] = 0;
            tmp[0+:32] = driver.dut.rb;
            tmp[addr*32+:32] = value;
            driver.dut.rb = tmp[0+:32];
        end
    endfunction

    function int read_reg_b;
        input int addr;
        logic [32-1:0] tmp;
        begin
            tmp[0+:32] = 0;
            tmp[0+:32] = driver.dut.rb;
            return tmp[addr*32+:32];
        end
    endfunction

    function void write_reg_y;
        input int value;
        input int addr;
        logic [31:0] tmp;
        begin
            tmp[0+:32] = 0;
            tmp[0+:32] = driver.dut.ry;
            tmp[addr*32+:32] = value;
            driver.dut.ry = tmp[0+:32];
        end
    endfunction

    function int read_reg_y;
        input int addr;
        logic [32-1:0] tmp;
        begin
            tmp[0+:32] = 0;
            tmp[0+:32] = driver.dut.ry;
            return tmp[addr*32+:32];
        end
    endfunction

    always_comb begin
        case(opcode)
            32'd0 : out = 32'hdeadbeef;
            32'd1 : begin
                case(id)
                    32'd0 : write_reg_a(in, addr);
                    32'd1 : write_reg_b(in, addr);
                    32'd2 : write_reg_y(in, addr);
                    default : $error("invalid id");
                endcase
            end
            32'd2 : begin
                case(id)
                    32'd0 : out = read_reg_a(addr);
                    32'd1 : out = read_reg_b(addr);
                    32'd2 : out = read_reg_y(addr);
                    default : $error("invalid id");
                endcase
            end
            default : $error("invalid opcode");
        endcase
    end

    adder dut (.clock(clock), .reset(reset));

endmodule
