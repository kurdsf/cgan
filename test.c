#include "gan.h"
#include "nn.h"
#include "utest.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <stdio.h>
#include <stdlib.h>

#define NN_ITERATIONS 10000
#define NN_MAX_AVG_ERR 0.01

static double vector_avg(const gsl_vector *x) {
  return gsl_vector_sum(x) / ((double)x->size);
}

UTEST(nn, learn_xor) {
  gsl_vector *input = gsl_vector_alloc(2);
  gsl_vector *label = gsl_vector_alloc(1);
  nn_t *nn = new_nn(2, 2, 1);
  for (size_t i = 0; i < NN_ITERATIONS; i++) {
    gsl_vector_set(input, 0, (double)(rand() % 2));
    gsl_vector_set(input, 0, (double)(rand() % 2));
    nn_forward(nn, input);
    gsl_vector_set(label, 0,
                   (double)(((int)gsl_vector_get(input, 0)) ^
                            ((int)gsl_vector_get(input, 1))));
    nn_backward(nn, label);
  }

  gsl_vector *errors = gsl_vector_alloc(10);
  for (size_t i = 0; i < 10; i++) {
    gsl_vector_set(input, 0, (double)(rand() % 2));
    gsl_vector_set(input, 0, (double)(rand() % 2));
    nn_forward(nn, input);
    gsl_vector_set(label, 0,
                   (double)(((int)gsl_vector_get(input, 0)) ^
                            ((int)gsl_vector_get(input, 1))));
    nn_backward(nn, label);
    gsl_vector_set(errors, i, nn_error(nn));
  }

  ASSERT_LT(vector_avg(errors), NN_MAX_AVG_ERR);

  gsl_vector_free(errors);
  gsl_vector_free(input);
  gsl_vector_free(label);
  nn_free(nn);
}

#define NN_PATH "nn_test"

UTEST(nn, io_routines_of_nn) {
  for (size_t i = 0; i < 10; i++) {
    nn_t *nn1 = new_nn(i, i / 2, i % 2);
    nn_write(NN_PATH, nn1);
    nn_t *nn2 = nn_read(NN_PATH);
    ASSERT_TRUE(gsl_matrix_equal(nn1->w1, nn2->w1));
    ASSERT_TRUE(gsl_matrix_equal(nn1->w2, nn2->w2));
    // we do not check if any other fields are equal,
    // for instance nn1->X_o, since these are uninitialized
    // because we haven't called nn_forward yet.
    nn_free(nn1);
    nn_free(nn2);
  }
  if (remove(NN_PATH) == -1) {
    perror("remove: " NN_PATH);
  }
}

// generate even numbers from 0 to 100.
void next_sample(gsl_vector *input) {
  gsl_vector_set(input, 0, (double)(2 * (rand() % 101)));
}

// the GAN should learn to generate even numbers
UTEST(gan, learn_even) {
  gan_t *gan = new_gan(1, &next_sample);
  gan_train(gan);
  gan_train(gan);
  // at least 90% of the generated numbers should be even.
  double nevens = 0;
  for (size_t i = 0; i < 10; i++) {
    gsl_vector *sample = gan_gen_sample(gan);
    int num = (int)(gsl_vector_get(sample, 0));
    if (num % 2 == 0)
      nevens++;
  }

  gan_free(gan);
  ASSERT_GT(nevens / 10, 0.9);
}

UTEST_MAIN();
