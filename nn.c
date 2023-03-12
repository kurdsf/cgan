#include "nn.h"
#include "mat.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void vec_ReLU(vec_t *O, const vec_t *X);
static void vec_softmax(vec_t *O, const vec_t *X);

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

  // intialize w1 and w2 with random values from [0.0, 1.0).
  // you may want to check out `man drand64`.
  for (size_t i = 0; i < (res->w1->m) * (res->w1->n); i++) {
    (res->w1->data)[i] = (scalar_t)drand48();
  }
  for (size_t i = 0; i < (res->w2->m) * (res->w2->n); i++) {
    (res->w2->data)[i] = (scalar_t)drand48();
  }

  return res;
}

void nn_train(nn_t* nn, const vec_t* inputs, const vec_t* labels) {
#ifndef NDEBUG
  assert(inputs->n == nn->w1->n);
  assert(labels->n == nn->w2->m);
#endif

  vec_t *X_h = new_vec(nn->w1->m);
  vec_t *O_h = new_vec(X_h->n);

  mat_vec_mul(X_h, nn->w1, inputs);
  vec_ReLU(O_h, X_h);

  vec_t *X_o = new_vec(nn->w2->m);
  vec_t *O_o = new_vec(X_o->n);

  mat_vec_mul(X_o, nn->w2, O_h);
  vec_softmax(O_o, X_o);

  puts("***************************************");
  for(size_t i = 0; i < (X_o->n); i++) {
          printf("%zu: %Lf.\n", i, (O_o->data)[i]);
  }
  puts("***************************************");

}


static void vec_ReLU(vec_t* O, const vec_t* X) {
#ifndef NDEBUG
        assert(O->n == X->n);
#endif 
        for(size_t i = 0; i < (X->n); i++) {
                scalar_t s = (X->data)[i];
                (O->data)[i] = (s > 0) ? s : 0;
        }
}


// numerically stable softmax
// approach taken from:
// https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
static void vec_softmax(vec_t* O, const vec_t *X) {
#ifndef NDEBUG
        assert(O->n == X->n);
#endif 

  // find the maximum of X
  scalar_t max = (X->data)[0];
  for (size_t i = 1; i < (X->n); i++) {
    if ((X->data)[i] > max)
      max = (X->data)[i];
  }

  for (size_t i = 0; i < (O->n); i++) {
    (O->data)[i] = (X->data)[i] - max;
  }

  scalar_t sum = 0;
  for (size_t i = 0; i < (O->n); i++) {
    sum += expl((O->data)[i]);
#ifndef NDEBUG
    if (errno != 0) {
      perror("softmax");
      exit(1);
    }
#endif
  }

  for (size_t i = 0; i < (O->n); i++) {
    (O->data)[i] = expl((O->data)[i]) / sum;
#ifndef NDEBUG
    if (errno != 0) {
      perror("softmax");
      exit(1);
    }
#endif
  }
}

void free_nn(nn_t *nn) {
  free_mat(nn->w1);
  free_mat(nn->w2);
  free(nn);
}
