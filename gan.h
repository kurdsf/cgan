#ifndef GAN_H
#define GAN_H

#include "nn.h"
#include <gsl/gsl_vector.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct {
  size_t isize;
  // write the next sample of the data set
  // to the vector inputs. The vector inputs
  // is already allocated to size isize, no need
  // to do that yourself.
  // shall return false if you just wrote the last sample of your
  // data set onto
  // inputs, else true.
  bool (*next_sample)(gsl_vector *sample);
  nn_t *gen;
  nn_t *dis;
} gan_t;

gan_t *new_gan(size_t isize, bool (*next_sample)(gsl_vector *sample);
// epochs is how often we iterate over the data set.
void gan_train(size_t epochs);
// ATTENTION: The vector this returns should not be freed.
// If you want to free it, you have to free the whole GAN via gan_free.
gsl_vector *gan_gen_sample(gan_t *gan);

void gan_free(gan_t *gan);
#endif // GAN_H
