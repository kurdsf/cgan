#include <assert.h>
#include <errno.h>
#include <gsl/gsl_vector.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

static bool get_next_image(FILE *f, gsl_vector *inputs, gsl_vector *labels) {
  static char *line = NULL;
  static size_t n = 0;

  ssize_t chars_read = getline(&line, &n, f);
  if (chars_read == -1) {
    if (errno != 0) {
      perror("get_next_image");
      exit(1);
    } else {
      // we may not read the last line for simplicity.
      free(line);
      return false;
    }
  }

  // remove the newline
  line[chars_read - 1] = '\0';

  char *slabel = strtok(line, ",");
  int label = atoi(slabel);
  // one-hot-encode labels.
  gsl_vector_set_basis(labels, label);

  char *spixel;
  size_t npixels = 0;
  while ((spixel = strtok(NULL, ",")) != NULL) {
    int pixel = atoi(spixel);
    gsl_vector_set(inputs, npixels, ((double)pixel) / 255.0);
    npixels++;
  }

  assert(npixels == 28 * 28);

  return true;
}

int main() {
  FILE *trainf = fopen("mnist_train.csv", "r");
  if (trainf == NULL) {
    perror("mnist_train.csv");
    return 1;
  }

  nn_t *nn = new_nn(28 * 28, 200, 10);

  gsl_vector *inputs = gsl_vector_alloc(28 * 28);
  gsl_vector *labels = gsl_vector_alloc(10);

  size_t nimg = 0;
  while (get_next_image(trainf, inputs, labels)) {
    nn_forward(nn, inputs);
    nn_backward(nn, labels);
    if (nimg % 100 == 0) {
      printf("debug: error: %f.\n", nn_error(nn));
    }
    nimg++;
  }

  gsl_vector_free(inputs);
  gsl_vector_free(labels);
  nn_free(nn);
  fclose(trainf);

  return 0;
}
