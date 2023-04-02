#include "nn.h"
#include <assert.h>
#include <errno.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline double ReLU(double x);
static inline double dReLU(double x);
static inline double sigmoid(double x);
static inline double dsigmoid(double x);
static void map_gsl_block(gsl_block *out, gsl_block *in, double (*f)(double));
static void mat_vec_mul(gsl_vector *out, const gsl_matrix *A,
                        const gsl_vector *x);

nn_t *new_nn(size_t isize, size_t hsize, size_t osize) {
  nn_t *res = malloc(sizeof(nn_t));
  if (res == NULL) {
    perror("new_nn-> malloc");
    exit(1);
  }

  _Static_assert(NFIELDS_OF_NN == 12, "new_nn expects NFIELDS_OF_NN to be 12");
  res->isize = isize;
  res->hsize = hsize;
  res->osize = osize;
  res->input = gsl_vector_alloc(isize);
  res->X_h = gsl_vector_alloc(hsize);
  res->O_h = gsl_vector_alloc(hsize);
  res->X_o = gsl_vector_alloc(osize);
  res->O_o = gsl_vector_alloc(osize);
  res->e1 = gsl_vector_alloc(osize);
  res->e2 = gsl_vector_alloc(hsize);
  res->w1 = gsl_matrix_alloc(hsize, isize);
  res->w2 = gsl_matrix_alloc(osize, hsize);

  srand48(time(NULL));

  for (size_t i = 0; i < (res->w1->block->size); i++) {
    (res->w1->block->data)[i] = drand48();
  }

  for (size_t i = 0; i < (res->w2->block->size); i++) {
    (res->w2->block->data)[i] = drand48();
  }

  return res;
}

void nn_forward(nn_t *nn, const gsl_vector *input) {
  assert(nn->isize == input->size);
  if (nn->input != input) {
    // save input for nn_backward.
    gsl_vector_memcpy(nn->input, input);
  }

  mat_vec_mul(nn->X_h, nn->w1, input);
  map_gsl_block(nn->O_h->block, nn->X_h->block, &ReLU);
  mat_vec_mul(nn->X_o, nn->w2, nn->O_h);
  map_gsl_block(nn->O_o->block, nn->X_o->block, &sigmoid);
}

void nn_backward(nn_t *nn, const gsl_vector *labels) {
  assert(nn->osize == labels->size);
  for (size_t i = 0; i < nn->osize; i++) {
    double diff = gsl_vector_get(labels, i) - gsl_vector_get(nn->O_o, i);
    gsl_vector_set(nn->e1, i, diff);
  }

  // e2 = transpose(w2) * e1
  gsl_blas_dgemv(CblasTrans, 1.0f, nn->w2, nn->e1, 0.0f, nn->e2);

  nn_backward_with_e1_and_e2_set(nn);
}

void nn_backward_with_e1_and_e2_set(nn_t *nn) {
  gsl_matrix_view input_T =
      gsl_matrix_view_vector(nn->input, 1, nn->input->size);
  gsl_matrix_view O_h_T = gsl_matrix_view_vector(nn->O_h, 1, nn->O_h->size);

  // w2 -= - LR * e1 * dsigmoid(O_o) * O_h_T <=> w2 = w2 + LR * e1 *
  // dsigmoid(O_o) * O_h_T
  gsl_matrix_view Y_1; // will be e1 * dsigmoid(O_o)
  gsl_vector *vec_Y_1 = gsl_vector_alloc(nn->O_o->size);
  map_gsl_block(vec_Y_1->block, nn->O_o->block, &dsigmoid);
  gsl_vector_mul(vec_Y_1, nn->e1);
  Y_1 = gsl_matrix_view_vector(vec_Y_1, vec_Y_1->size, 1);

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, LR, &(Y_1.matrix), &(O_h_T.matrix),
                 1.0, nn->w2);
  gsl_vector_free(vec_Y_1);

  // w1 -= - LR * e2 * dsigmoid(O_h) * input_T <=> w1 = w1 + LR * e2 *
  // dsigmoid(O_h) * input_T
  gsl_matrix_view Y_2; // will be e2 * dsigmoid(O_h)
  gsl_vector *vec_Y_2 = gsl_vector_alloc(nn->O_h->size);
  map_gsl_block(vec_Y_2->block, nn->O_h->block, &dReLU);
  gsl_vector_mul(vec_Y_2, nn->e2);
  Y_2 = gsl_matrix_view_vector(vec_Y_2, vec_Y_2->size, 1);

  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, LR, &(Y_2.matrix),
                 &(input_T.matrix), 1.0, nn->w1);
  gsl_vector_free(vec_Y_2);
}

double nn_error(nn_t *nn) {
  double err = 0.0F;
  for (size_t i = 0; i < nn->e1->size; i++) {
    err += gsl_vector_get(nn->e1, i) * gsl_vector_get(nn->e1, i);
  }
  err /= nn->e1->size;
  return err;
}

void nn_write(const char *path, const nn_t *nn) {
  _Static_assert(NFIELDS_OF_NN == 12,
                 "nn_write expects NFIELDS_OF_NN to be 12");
  FILE *f = fopen(path, "w");
  if (f == NULL) {
    fprintf(stderr, "nn_write: %s: %s\n", path, strerror(errno));
    exit(1);
  }

  size_t items_written;
  items_written = fwrite(&(nn->isize), sizeof(double), 1, f);
  if (items_written != 1) {
    fprintf(stderr, "nn_write: %s: %s\n", path, strerror(errno));
    exit(1);
  }
  items_written = fwrite(&(nn->hsize), sizeof(double), 1, f);
  if (items_written != 1) {
    fprintf(stderr, "nn_write: %s: %s\n", path, strerror(errno));
    exit(1);
  }
  items_written = fwrite(&(nn->osize), sizeof(double), 1, f);
  if (items_written != 1) {
    fprintf(stderr, "nn_write: %s: %s\n", path, strerror(errno));
    exit(1);
  }

  if (gsl_vector_fwrite(f, nn->input) == GSL_EFAILED) {
    fprintf(stderr, "nn_write: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fwrite(f, nn->X_h) == GSL_EFAILED) {
    fprintf(stderr, "nn_write: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fwrite(f, nn->O_h) == GSL_EFAILED) {
    fprintf(stderr, "nn_write: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fwrite(f, nn->X_o) == GSL_EFAILED) {
    fprintf(stderr, "nn_write: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fwrite(f, nn->O_o) == GSL_EFAILED) {
    fprintf(stderr, "nn_write: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fwrite(f, nn->e1) == GSL_EFAILED) {
    fprintf(stderr, "nn_write: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fwrite(f, nn->e2) == GSL_EFAILED) {
    fprintf(stderr, "nn_write: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_matrix_fwrite(f, nn->w1) == GSL_EFAILED) {
    fprintf(stderr, "nn_write: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_matrix_fwrite(f, nn->w2) == GSL_EFAILED) {
    fprintf(stderr, "nn_write: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }

  fclose(f);
}

nn_t *nn_read(const char *path) {
  _Static_assert(NFIELDS_OF_NN == 12, "nn_read expects NFIELDS_OF_NN to be 12");
  FILE *f = fopen(path, "r");
  if (f == NULL) {
    fprintf(stderr, "nn_read: %s: %s\n", path, strerror(errno));
    exit(1);
  }

  size_t isize, hsize, osize;
  size_t items_read;
  items_read = fread(&isize, sizeof(double), 1, f);
  if (items_read != 1) {
    fprintf(stderr, "nn_read: %s: %s\n", path, strerror(errno));
    exit(1);
  }
  items_read = fread(&hsize, sizeof(double), 1, f);
  if (items_read != 1) {
    fprintf(stderr, "nn_read: %s: %s\n", path, strerror(errno));
    exit(1);
  }
  items_read = fread(&osize, sizeof(double), 1, f);
  if (items_read != 1) {
    fprintf(stderr, "nn_read: %s: %s\n", path, strerror(errno));
    exit(1);
  }

  nn_t *nn = new_nn(isize, hsize, osize);

  if (gsl_vector_fread(f, nn->input) == GSL_EFAILED) {
    fprintf(stderr, "nn_read: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fread(f, nn->X_h) == GSL_EFAILED) {
    fprintf(stderr, "nn_read: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fread(f, nn->O_h) == GSL_EFAILED) {
    fprintf(stderr, "nn_read: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fread(f, nn->X_o) == GSL_EFAILED) {
    fprintf(stderr, "nn_read: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fread(f, nn->O_o) == GSL_EFAILED) {
    fprintf(stderr, "nn_read: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fread(f, nn->e1) == GSL_EFAILED) {
    fprintf(stderr, "nn_read: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_vector_fread(f, nn->e2) == GSL_EFAILED) {
    fprintf(stderr, "nn_read: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_matrix_fread(f, nn->w1) == GSL_EFAILED) {
    fprintf(stderr, "nn_read: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }
  if (gsl_matrix_fread(f, nn->w2) == GSL_EFAILED) {
    fprintf(stderr, "nn_read: %s: %s\n", path, gsl_strerror(GSL_EFAILED));
  }

  fclose(f);
  return nn;
}

static inline double ReLU(double x) { return (x > 0) ? x : 0; }

static inline double dReLU(double x) { return (x > 0) ? 1 : 0; }

static inline double sigmoid(double x) { return 1 / (1 + exp(-x)); }
static inline double dsigmoid(double x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

static void map_gsl_block(gsl_block *out, gsl_block *in, double (*f)(double)) {
  assert(out->size == in->size);
  for (size_t i = 0; i < out->size; i++) {
    (out->data)[i] = f((in->data)[i]);
  }
}

static void mat_vec_mul(gsl_vector *out, const gsl_matrix *A,
                        const gsl_vector *x) {
  gsl_blas_dgemv(CblasNoTrans, 1.0f, A, x, 0.0f, out);
}

void nn_free(nn_t *nn) {
  _Static_assert(NFIELDS_OF_NN == 12, "nn_free expects NFIELDS_OF_NN to be 12");
  gsl_vector_free(nn->input);
  gsl_vector_free(nn->X_h);
  gsl_vector_free(nn->O_h);
  gsl_vector_free(nn->X_o);
  gsl_vector_free(nn->O_o);
  gsl_vector_free(nn->e1);
  gsl_vector_free(nn->e2);
  gsl_matrix_free(nn->w1);
  gsl_matrix_free(nn->w2);
  free(nn);
}
