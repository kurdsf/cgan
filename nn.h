#ifndef NN_H
#define NN_H

#include <stddef.h>

#include "mat.h"

// the learning rate.
// change and recompile as needed.
#define LR 0.001L

typedef struct {
  mat_t *w1;
  mat_t *w2;
} nn_t;

nn_t *new_nn(size_t isize, size_t hsize, size_t osize);
// Note that this function logs some infos to stdout.
void nn_train(nn_t *nn, const vec_t *inputs, const vec_t *labels);
void free_nn(nn_t *nn);
void nn_write(const char *path, const nn_t *nn);
// ATTENTION:
// This function DOES NOT PERFORM ANY KIND OF SANITY CHECKS.
nn_t *nn_read(const char *path);

#endif // NN_H
