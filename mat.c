#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "mat.h"

mat_t *new_mat(size_t m, size_t n) {
  mat_t *res = malloc(sizeof(mat_t));
  // In our program we cannot recover from out-of-memory errors,
  // so we will just exit.
#ifndef NDEBUG
  if (res == NULL) {
    perror("new_mat");
    exit(1);
  }
#endif

  scalar_t *data = malloc(sizeof(scalar_t) * n * m);
#ifndef NDEBUG
  if (data == NULL) {
    perror("new_mat");
    exit(1);
  }
#endif

  res->m = m;
  res->n = n;
  res->data = data;

  return res;
}

void free_mat(mat_t *A) {
  free(A->data);
  free(A);
}

vec_t *new_vec(size_t n) {
  vec_t *res = malloc(sizeof(vec_t));
#ifndef NDEBUG
  if (res == NULL) {
    perror("new_vec");
    exit(1);
  }
#endif

  scalar_t *data = malloc(sizeof(scalar_t) * n);
#ifndef NDEBUG
  if (data == NULL) {
    perror("new_vec");
    exit(1);
  }
#endif

  res->n = n;
  res->data = data;

  return res;
}

void free_vec(vec_t *x) {
  free(x->data);
  free(x);
}

void mat_vec_mul(vec_t *restrict dest, const mat_t *restrict A,
                 const vec_t *restrict x) {
#ifndef NDEBUG
  assert(dest->n == A->m);
  assert(x->n == A->n);
#endif
  for (size_t i = 0; i < A->m; i++) {
    scalar_t res = 0;
    for (size_t j = 0; j < A->n; j++) {
      res += (x->data)[j] * (A->data)[A->m * i + j];
    }
    (dest->data)[i] = res;
  }
}

