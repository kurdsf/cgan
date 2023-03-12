#ifndef NN_H
#define NN_H

#include <stddef.h>

#include "mat.h"

// the learning rate.
// change and recompile as needed.
#define LR 0.0000001L

typedef struct {
  mat_t *w1;
  mat_t *w2;
} nn_t;

nn_t *new_nn(size_t isize, size_t hsize, size_t osize);
// Note that this function logs some infos to stdout.
void nn_train(nn_t* nn, const vec_t* inputs, const vec_t* labels);
void free_nn(nn_t *nn);

#endif // NN_H
