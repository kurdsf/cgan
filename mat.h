#ifndef MAT_H
#define MAT_H

#include <stddef.h>

typedef long double scalar_t;

typedef struct {
  scalar_t *data;
  size_t m, n;
} mat_t;

typedef struct {
  scalar_t *data;
  size_t n;
} vec_t;

mat_t *new_mat(size_t m, size_t n);
void free_mat(mat_t *A);

vec_t *new_vec(size_t n);
void free_vec(vec_t *x);

void mat_vec_mul(vec_t *restrict dest, const mat_t *restrict A,
                 const vec_t *restrict x);

#endif // MAT_H
