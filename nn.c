#include "nn.h"
#include <assert.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

static void map_gsl_block(gsl_block out, gsl_block in, double (*f)(double));

nn_t new_nn(size_t isize, size_t hsize, size_t osize) {
  nn_t res;
  res.isize = isize;
  res.hsize = hsize;
  res.osize = osize;
  res.X_h = gsl_vector_alloc(hsize);
  res.O_h = gsl_vector_alloc(hsize);
  res.X_o = gsl_vector_alloc(osize);
  res.O_o = gsl_vector_alloc(osize);
  res.e1 = gsl_vector_alloc(osize);
  res.e2 = gsl_vector_alloc(hsize);
  res.w1 = gsl_matrix_alloc(isize, hsize);
  res.w2 = gsl_matrix_alloc(hsize, isize);

  srand48(time(NULL));

  for (size_t i = 0; i < (res.w1->block->size); i++) {
    (res.w1->block->data)[i] = drand48();
  }

  for (size_t i = 0; i < (res.w2->block->size); i++) {
    (res.w2->block->data)[i] = drand48();
  }

  return res;
}

static void map_gsl_block(gsl_block out, gsl_block in, double (*f)(double)) {
  assert(out.size == in.size);
  for (size_t i = 0; i < out.size; i++) {
    out.data[i] = f(in.data[i]);
  }
}

void nn_free(nn_t nn) {
  gsl_vector_free(nn.X_h);
  gsl_vector_free(nn.O_h);
  gsl_vector_free(nn.X_o);
  gsl_vector_free(nn.O_o);
  gsl_vector_free(nn.e1);
  gsl_vector_free(nn.e2);
  gsl_matrix_free(nn.w1);
  gsl_matrix_free(nn.w2);
}
