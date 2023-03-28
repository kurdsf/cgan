#include <assert.h>
#include <gsl/gsl_vector.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "nn.h"

static bool get_next_image(gsl_vector *inputs, gsl_vector *labels) {
  static FILE *finputs = NULL;
  static FILE *flabels = NULL;
  static unsigned char raw_inputs[28 * 28];
  static unsigned char raw_labels[1];

  if (finputs == NULL) {
    finputs = fopen("train-images", "r");
    if (finputs == NULL) {
      perror("train-images");
      exit(1);
    }
  }

  if (flabels == NULL) {
    flabels = fopen("train-labels", "r");
    if (finputs == NULL) {
      perror("train-labels");
      exit(1);
    }
  }

  ssize_t inputs_read =
      fread(raw_inputs, sizeof(unsigned char), 28 * 28, finputs);
  if (inputs_read != 28 * 28) {
    perror("train-images: fread");
    exit(1);
  }

  ssize_t labels_read = fread(raw_labels, sizeof(unsigned char), 1, flabels);
  if (labels_read != 1) {
    perror("train-labels: fread");
    exit(1);
  }

  assert(inputs->size == 28 * 28);
  // we will one hot-encode labels with raw_labels.
  assert(labels->size == 10);

  for (size_t i = 0; i < 28 * 28; i++) {
    // the inputs should be between 0.0 and 1.0, thus we divide by 255.0.
    gsl_vector_set(inputs, i, ((double)raw_inputs[i]) / 255.0);
  }

  gsl_vector_set_basis(labels, raw_labels[0]);

  int fi = fgetc(finputs);
  int fl = fgetc(flabels);

  ungetc(fi, finputs);
  ungetc(fl, flabels);

  if (fi == EOF) {
    assert(fl == EOF);
    return false;
  }

  return true;
}

int main() {
  nn_t *nn = new_nn(28 * 28, 200, 10);

  gsl_vector *inputs = gsl_vector_alloc(28 * 28);
  gsl_vector *labels = gsl_vector_alloc(10);

  get_next_image(inputs, labels);
  get_next_image(inputs, labels);

  puts("inputs:");
  gsl_vector_fprintf(stdout, inputs, "%f");
  puts("labels:");
  gsl_vector_fprintf(stdout, labels, "%f");

  gsl_vector_free(inputs);
  gsl_vector_free(labels);
  nn_free(nn);

  return 0;
}
