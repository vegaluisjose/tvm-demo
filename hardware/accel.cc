#include "accel.h"
#include "driver.h"
#include <stdint.h>

void bias_add(int *a, int *b, int *out, int out_dim, int in_dim) {
  DriverHandle dev = DriverAlloc();
  DriverReset(dev, 1);
  for (int64_t i = 0; i < out_dim; ++i) {
    for (int64_t j = 0; j < in_dim; ++j) {
      int64_t k = i * in_dim + j;
      DriverWrite(dev, 0, 0, a[k]);
      DriverWrite(dev, 1, 0, b[j]);
      DriverRun(dev, 1);
      out[k] = DriverRead(dev, 2, 0);
    }
  }
  DriverDealloc(dev);
}