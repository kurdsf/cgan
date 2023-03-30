#include "gan.h"
#include "nn.h"
#include <gsl/gsl_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

gan_t *new_gan(size_t isize, bool (*next_sample)(gsl_vector *sample)) {
  gan_t *res = malloc(sizeof(gan_t));
  if (res == NULL) {
    perror("new_gan");
    exit(1);
  }

  res->next_sample = next_sample;
  res->gen = new_nn(isize, isize / 2, isize);
  res->dis = new_nn(isize, isize, 1);
  return res;
}

// TODO: Implement gan_train

gsl_vector *gan_gen_sample(gan_t *gan) {
  for (size_t i = 0; i < (gan->gen->isize); i++) {
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
