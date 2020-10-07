#include <stdint.h>
#include "accel.h"

void bias_add(int* a, int* b, int* out, int out_dim, int in_dim) {
    for (int64_t i = 0; i < out_dim; ++i) {
        for (int64_t j = 0; j < in_dim; ++j) {
            int64_t k = i * in_dim + j;
            out[k] = a[k] + b[j];
        }
    }
}
