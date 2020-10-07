#include <stdint.h>
#include "accel.h"
#include "device.h"

void bias_add(int* a, int* b, int* out, int out_dim, int in_dim) {
    DeviceHandle dev = DeviceAlloc();
    DeviceReset(dev, 1);
    for (int64_t i = 0; i < out_dim; ++i) {
        for (int64_t j = 0; j < in_dim; ++j) {
            int64_t k = i * in_dim + j;
            DeviceWrite(dev, 0, 0, a[k]);
            DeviceWrite(dev, 1, 0, b[j]);
            DeviceRun(dev, 1);
            out[k] = DeviceRead(dev, 2, 0);
        }
    }
    DeviceDealloc(dev);
}