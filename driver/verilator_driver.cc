#include "Top.h"
#include "driver.h"

vluint64_t main_time = 0;

double sc_time_stamp() { return main_time; }

DriverHandle DriverAlloc() {
  Top *top = new Top;
  return static_cast<DriverHandle>(top);
}

void DriverDealloc(DriverHandle handle) { delete static_cast<Top *>(handle); }

int DriverRead(DriverHandle handle, int id, int addr) {
  Top *top = static_cast<Top *>(handle);
  top->opcode = 2;
  top->id = id;
  top->addr = addr;
  top->eval();
  return top->out;
}

void DriverWrite(DriverHandle handle, int id, int addr, int value) {
  Top *top = static_cast<Top *>(handle);
  top->opcode = 1;
  top->id = id;
  top->addr = addr;
  top->in = value;
  top->eval();
}

void DriverReset(DriverHandle handle, int n) {
  Top *top = static_cast<Top *>(handle);
  top->clock = 0;
  top->reset = 1;
  main_time = 0;
  while (!Verilated::gotFinish() &&
         main_time < static_cast<vluint64_t>(n * 10)) {
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

void DriverRun(DriverHandle handle, int n) {
  Top *top = static_cast<Top *>(handle);
  top->clock = 0;
  main_time = 0;
  while (!Verilated::gotFinish() &&
         main_time < static_cast<vluint64_t>(n * 10)) {
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
