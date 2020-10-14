#include "device.h"
#include "stdio.h"

int main() {
  DeviceHandle dev = DeviceAlloc();
  DeviceReset(dev, 3);
  DeviceWrite(dev, 0, 0, 4);
  DeviceWrite(dev, 1, 0, 9);
  DeviceRun(dev, 1);
  printf("result:%d\n", DeviceRead(dev, 2, 0));
  DeviceDealloc(dev);
  return 0;
}
