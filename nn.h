#ifndef NN_H
#define NN_H

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <stddef.h>

// the learning rate.
// change and recompile as needed.
#define LR 0.1L

// if we add an field to nn_t, we have to modify
// nn_write and nn_read. Thus these functions check if
// NFIELDS_OF_NN is the value they expect.
#define NFIELDS_OF_NN 12

typedef struct {
  size_t isize;
  size_t osize;
  size_t hsize;
  gsl_vector *input;
  gsl_vector *X_h;
  gsl_vector *O_h;
  gsl_vector *X_o;
  gsl_vector *O_o;
  gsl_vector *e1;
  gsl_vector *e2;
  gsl_matrix *w1;
  gsl_matrix *w2;
} nn_t;

nn_t *new_nn(size_t isize, size_t hsize, size_t osize);
void nn_forward(nn_t *nn, const gsl_vector *input);
void nn_backward(nn_t *nn, const gsl_vector *label);
void nn_backward_with_e1_and_e2_set(nn_t *nn);
double nn_error(nn_t *nn);

void nn_free(nn_t *nn);
void nn_write(const char *path, const nn_t *nn);
// ATTENTION:
// This function DOES NOT PERFORM ANY KIND OF SANITY CHECKS.
nn_t *nn_read(const char *path);

#endif // NN_H
