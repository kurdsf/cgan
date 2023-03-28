#include "nn.h"
#include <assert.h>
#include <errno.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void map_gsl_block(gsl_block out, gsl_block in, double (*f)(double));
static void mat_vec_mul(gsl_vector *out, const gsl_matrix *A,
                        const gsl_vector *x);

nn_t *new_nn(size_t isize, size_t hsize, size_t osize) {
  nn_t *res = malloc(sizeof(nn_t));
  if (res == NULL) {
    perror("new_nn-> malloc");
    exit(1);
  }

  res->isize = isize;
  res->hsize = hsize;
  res->osize = osize;
  res->X_h = gsl_vector_alloc(hsize);
  res->O_h = gsl_vector_alloc(hsize);
  res->X_o = gsl_vector_alloc(osize);
  res->O_o = gsl_vector_alloc(osize);
  res->e1 = gsl_vector_alloc(osize);
  res->e2 = gsl_vector_alloc(hsize);
  res->w1 = gsl_matrix_alloc(isize, hsize);
  res->w2 = gsl_matrix_alloc(hsize, isize);

  srand48(time(NULL));

  for (size_t i = 0; i < (res->w1->block->size); i++) {
    (res->w1->block->data)[i] = drand48();
  }

  for (size_t i = 0; i < (res->w2->block->size); i++) {
    (res->w2->block->data)[i] = drand48();
  }

  return res;
}

void nn_write(const char *path, const nn_t *nn) {
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

static void map_gsl_block(gsl_block out, gsl_block in, double (*f)(double)) {
  assert(out.size == in.size);
  for (size_t i = 0; i < out.size; i++) {
    out.data[i] = f(in.data[i]);
  }
}

static void mat_vec_mul(gsl_vector *out, const gsl_matrix *A,
                        const gsl_vector *x) {
  gsl_blas_dgemv(CblasNoTrans, 1.0f, A, x, 0.0f, out);
}

void nn_free(nn_t *nn) {
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
