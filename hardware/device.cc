#include "device.h"
#include "Top.h"

vluint64_t main_time = 0;

double sc_time_stamp() { return main_time; }

DeviceHandle DeviceAlloc() {
    Top* top = new Top;
    return static_cast<DeviceHandle>(top);
}

void DeviceDealloc(DeviceHandle handle) {
    delete static_cast<Top*>(handle);
}

int DeviceRead(DeviceHandle handle, int id, int addr) {
    Top* top = static_cast<Top*>(handle);
    top->opcode = 2;
    top->id = id;
    top->addr = addr;
    top->eval();
    return top->out;
}

void DeviceWrite(DeviceHandle handle, int id, int addr, int value) {
    Top* top = static_cast<Top*>(handle);
    top->opcode = 1;
    top->id = id;
    top->addr = addr;
    top->in = value;
    top->eval();
}

void DeviceReset(DeviceHandle handle, int n) {
    Top* top = static_cast<Top*>(handle);
    top->clock = 0;
    top->reset = 1;
    main_time = 0;
    while (!Verilated::gotFinish() && main_time < static_cast<vluint64_t>(n*10)) {
        if ((main_time % 10) == 1) {
            top->clock = 1;
        }
        if ((main_time % 10) == 6) {
            top->reset = 0;
        }
        top->eval();
        main_time++;
    }
    top->reset = 0;
}


void DeviceRun(DeviceHandle handle, int n) {
    Top* top = static_cast<Top*>(handle);
    top->clock = 0;
    main_time = 0;
    while (!Verilated::gotFinish() && main_time < static_cast<vluint64_t>(n*10)) {
        if ((main_time % 10) == 1) {
            top->clock = 1;
        }
        if ((main_time % 10) == 6) {
            top->clock = 0;
        }
        top->eval();
        main_time++;
    }
}
