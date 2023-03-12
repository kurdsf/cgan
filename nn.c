#include "nn.h"
#include "mat.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void vec_ReLU(vec_t *O, const vec_t *X);
static scalar_t ReLU_diff(scalar_t x);


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
  vec_ReLU(O_o, X_o);

  vec_t* e1 = new_vec(O_o->n);
  scalar_t err = 0;
  for(size_t i=0; i < (e1->n); i++) {
          scalar_t diff = (O_o->data)[i] - (labels->data)[i];
          (e1->data)[i] = diff * diff;
          err += (e1->data)[i];
  }

  printf("error value: %Lf.\n", err);

  
  for(size_t i = 0; i < (nn->w2->m); i++) {
          for(size_t j = 0; j < (nn->w2->n); j++) {
                  (nn->w2->data)[(nn->w2->m) * i + j] -= LR * (-2) *((labels->data)[i] - ((O_o->data)[i])) * ReLU_diff((O_o->data)[i]) * (O_h->data)[j];
          }
  }

  free_vec(X_h);
  free_vec(O_h);
  free_vec(X_o);
  free_vec(O_o);
  free_vec(e1);
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

static scalar_t ReLU_diff(scalar_t x) {
        // we return 0.001 instead of 0.0 so 
        // the whole term above is not zero.
        return (x > 0.0) ? 1.0 : 0.001;
}

void free_nn(nn_t *nn) {
  free_mat(nn->w1);
  free_mat(nn->w2);
  free(nn);
}
