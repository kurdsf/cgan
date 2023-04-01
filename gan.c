#include "gan.h"
#include "nn.h"
#include <assert.h>
#include <errno.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_vector.h>
#include <stdio.h>
#include <stdlib.h>

static void train_dis(gan_t *gan);
static void get_real_or_fake_sample(gan_t *gan, double *is_real);
static void train_gen(gan_t *gan);

gan_t *new_gan(size_t isize, void (*next_sample)(gsl_vector *)) {
  gan_t *res = malloc(sizeof(gan_t));
  if (res == NULL) {
    perror("new_gan");
    exit(1);
  }

  assert(isize != 0);
  res->isize = isize;
  res->next_sample = next_sample;
  res->gen = new_nn(isize, isize / 2 + 1, isize);
  res->dis = new_nn(isize, isize, 1);
  return res;
}

void gan_train(gan_t *gan) {
  train_dis(gan);
  train_gen(gan);
}

static void train_dis(gan_t *gan) {
  double is_real;
  gsl_vector *label = gsl_vector_alloc(1);
  for (size_t i = 0; i < DIS_TRAINING_NSAMPLES; i++) {
    get_real_or_fake_sample(gan, &is_real);
    nn_forward(gan->dis, gan->dis->input);
    gsl_vector_set(label, 0, is_real);
    nn_backward(gan->dis, label);
  }
  gsl_vector_free(label);
}

static void train_gen(gan_t *gan) {
  for (size_t i = 0; i < GEN_TRAINING_NSAMPLES; i++) {
    gsl_vector_memcpy(gan->dis->input, gan_gen_sample(gan));
    nn_forward(gan->dis, gan->dis->input);
    // we do nn_backward by hand here.
    gsl_vector_set(gan->dis->e1, 0, 0.0 - gsl_vector_get(gan->dis->O_o, 0));
    gsl_blas_dgemv(CblasTrans, 1.0f, gan->dis->w2, gan->dis->e1, 0.0f,
                   gan->dis->e2);
    gsl_blas_dgemv(CblasTrans, 1.0f, gan->dis->w2, gan->dis->e2, 0.0f,
                   gan->gen->e1);
    gsl_blas_dgemv(CblasTrans, 1.0f, gan->gen->w2, gan->gen->e2, 0.0f,
                   gan->gen->e1);
    nn_backward_with_e1_and_e2_set(gan->gen);
  }
}

static void get_real_or_fake_sample(gan_t *gan, double *is_real) {
  *is_real = (double)(rand() % 2);
  if (*is_real == 0) {
    gan->next_sample(gan->dis->input);
  } else {
    gsl_vector_memcpy(gan->dis->input, gan_gen_sample(gan));
  }
}

gsl_vector *gan_gen_sample(gan_t *gan) {
  for (size_t i = 0; i < (gan->isize); i++) {
    if (gan->gen->input->size <= i) {
      puts("gotcha");
      exit(1);
    }
    gsl_vector_set(gan->gen->input, i, drand48());
  }
  nn_forward(gan->gen, gan->gen->input);

  return gan->gen->O_o;
}

void gan_free(gan_t *gan) {
  nn_free(gan->gen);
  nn_free(gan->dis);
  free(gan);
}
