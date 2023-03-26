#include "nn.h"
#include "mat.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static size_t efread(void *ptr, size_t size, size_t nmemb, FILE *stream);
static size_t efwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);

static scalar_t sigmoid(scalar_t x);
static scalar_t dsigmoid(scalar_t x);

nn_t *new_nn(size_t isize, size_t hsize, size_t osize) {
  nn_t *res = malloc(sizeof(nn_t));
#ifndef NDEBUG
  if (res == NULL) {
    perror("new_nn");
    exit(1);
  }
#endif

  res->w1 = new_mat(hsize, isize);
  res->w2 = new_mat(osize, hsize);

  srand48(time(NULL));

  // intialize w1 and w2 with random values from [-1.0, 1.0).
  // you may want to check out `man drand64`.
  for (size_t i = 0; i < (res->w1->m) * (res->w1->n); i++) {
    (res->w1->data)[i] = ((scalar_t)drand48()) * 2.0L - 1.0L;
  }
  for (size_t i = 0; i < (res->w2->m) * (res->w2->n); i++) {
    (res->w2->data)[i] = ((scalar_t)drand48()) * 2.0L - 1.0L;
  }

  return res;
}

scalar_t nn_train(nn_t *nn, const vec_t *inputs, const vec_t *labels) {
#ifndef NDEBUG
  assert(inputs->n == nn->w1->n);
  assert(labels->n == nn->w2->m);
#endif

  vec_t *X_h = new_vec(nn->w1->m);
  vec_t *O_h = new_vec(X_h->n);

  mat_vec_mul(X_h, nn->w1, inputs);

  // apply activation function
  for (size_t i = 0; i < (X_h->n); i++) {
    (O_h->data)[i] = sigmoid((X_h->data)[i]);
  }

  vec_t *X_o = new_vec(nn->w2->m);
  vec_t *O_o = new_vec(X_o->n);

  mat_vec_mul(X_o, nn->w2, O_h);

  // apply activation function
  for (size_t i = 0; i < (X_o->n); i++) {
    (O_o->data)[i] = sigmoid((X_o->data)[i]);
  }

  scalar_t err = 0;

  vec_t *e1 = new_vec(O_o->n);
  for (size_t i = 0; i < (e1->n); i++) {
    (e1->data)[i] = (labels->data)[i] - (O_o->data)[i];
    err += (e1->data)[i] * (e1->data)[i];
  }

  for (size_t i = 0; i < (nn->w2->m); i++) {
    for (size_t j = 0; j < (nn->w2->n); j++) {
      // The individual derivates
      scalar_t d_err_d_O_o_i = -2 * (e1->data)[i];
      scalar_t d_O_o_i_d_w2_i_j = dsigmoid((X_o->data)[i]) * (O_h->data)[j];

      (nn->w2->data)[(nn->w2->m) * i + j] -=
          LR * d_err_d_O_o_i * d_O_o_i_d_w2_i_j;
    }
  }

  // backpropagate the error
  vec_t *e2 = new_vec(O_h->n);

  for (size_t i = 0; i < (e2->n); i++) {
    (e2->data)[i] = 0.0L;
    for (size_t j = 0; j < (e1->n); j++) {
      (e2->data)[i] += (e1->data)[j] * (nn->w2->data)[(nn->w2->m) * j + i];
    }
  }

  for (size_t i = 0; i < (nn->w1->m); i++) {
    for (size_t j = 0; j < (nn->w1->n); j++) {
      // The individual derivates
      scalar_t d_err_O_h_i = -2 * (e2->data)[i];
      scalar_t d_O_h_i_d_w1_i_j = dsigmoid((X_h->data)[i]) * (inputs->data)[j];

      (nn->w1->data)[(nn->w1->m) * i + j] -=
          LR * d_err_O_h_i * d_O_h_i_d_w1_i_j;
    }
  }

  free_vec(X_h);
  free_vec(O_h);
  free_vec(X_o);
  free_vec(O_o);
  free_vec(e1);
  free_vec(e2);

  return err;
}

static scalar_t sigmoid(scalar_t x) {
  scalar_t res = 1 / (1 + expl(-x));

#ifndef NDEBUG
  if (errno != 0) {
    perror("sigmoid");
    fprintf(stderr, "debug: the value was: %Lf", res);
    exit(1);
  }
#endif
  return res;
}

static scalar_t dsigmoid(scalar_t x) { return sigmoid(x) * (1 - sigmoid(x)); }

void free_nn(nn_t *nn) {
  free_mat(nn->w1);
  free_mat(nn->w2);
  free(nn);
}

static size_t efread(void *ptr, size_t size, size_t nmemb, FILE *stream) {
  size_t items_read = fread(ptr, size, nmemb, stream);
  if (items_read != nmemb) {
    perror("nn_read: fread");
    exit(1);
  }
  return items_read;
}

static size_t efwrite(const void *ptr, size_t size, size_t nmemb,
                      FILE *stream) {
  size_t items_written = fwrite(ptr, size, nmemb, stream);
  if (items_written != nmemb) {
    perror("nn_write: fwrite");
    exit(1);
  }
  return items_written;
}

nn_t *nn_read(const char *path) {
  nn_t *res = malloc(sizeof(nn_t));

  if (res == NULL) {
    fprintf(stderr, "nn_read: %s: %s.\n", path, strerror(errno));
    exit(1);
  }

  FILE *f = fopen(path, "r");
  if (f == NULL) {
    fprintf(stderr, "nn_read: %s: %s.\n", path, strerror(errno));
    exit(1);
  }

  size_t i, h, o;
  efread(&i, sizeof(size_t), 1, f);
  efread(&h, sizeof(size_t), 1, f);
  efread(&o, sizeof(size_t), 1, f);

  mat_t *w1 = new_mat(h, i);
  mat_t *w2 = new_mat(o, h);

  efread(w1->data, sizeof(scalar_t), h * i, f);
  efread(w1->data, sizeof(scalar_t), o * h, f);

  fclose(f);

  res->w1 = w1;
  res->w2 = w2;

  return res;
}

void nn_write(const char *path, const nn_t *nn) {
  FILE *f = fopen(path, "w");
  if (f == NULL) {
    fprintf(stderr, "nn_write: %s: %s.\n", path, strerror(errno));
    exit(1);
  }

  efwrite(&(nn->w1->n), sizeof(size_t), 1, f);
  efwrite(&(nn->w1->m), sizeof(size_t), 1, f);
  efwrite(&(nn->w2->m), sizeof(size_t), 1, f);

  efwrite(nn->w1->data, sizeof(scalar_t), (nn->w1->m) * (nn->w1->n), f);
  efwrite(nn->w2->data, sizeof(scalar_t), (nn->w2->m) * (nn->w2->n), f);

  fclose(f);
}
