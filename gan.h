#ifndef GAN_H
#define GAN_H

#include "nn.h"
#include <gsl/gsl_vector.h>
#include <stddef.h>

#define GEN_TRAINING_NSAMPLES 200
#define DIS_TRAINING_NSAMPLES 200
#define GAN_NOISE_MULTIPLIER 400

typedef struct {
  size_t isize;
  // write the next sample of the data set
  // to the vector inputs. The vector argument
  // is already allocated to size isize, no need
  // to do that yourself.
  // you most likely don't have an infinite number
  // of samples, so if the samples in your data set are
  // exhausted, just start from the beginning.
  void (*next_sample)(gsl_vector *);
  nn_t *gen;
  nn_t *dis;
} gan_t;

gan_t *new_gan(size_t isize, void (*next_sample)(gsl_vector *));

// you will have to call
// this method multiple times to get a good GAN.
void gan_train(gan_t *gan);
// ATTENTION: The vector this returns should not be freed.
// If you want to free it, you have to free the whole GAN via gan_free.
gsl_vector *gan_gen_sample(gan_t *gan);
void gan_free(gan_t *gan);
#endif // GAN_H
