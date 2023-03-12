#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mat.h"
#include "nn.h"

// Returns NULL incase of EOF.
// Furthermore, it expects '\n' instead of '\r\n'
vec_t *get_next_image(FILE *f, int *label) {
  static char *line;
  static size_t n;
  // check if we have reached EOF yet.
  int c;
  if ((c = fgetc(f)) == EOF) {
    free(line);
    return NULL;
  } else {
    ungetc(c, f);
  }

  ssize_t errc = getline(&line, &n, f);
  if (errc == (ssize_t)-1) {
    // If errno == 0, we reached EOF.
    // This means we are at the last image or line
    // to process, we just continue as normal
    // and in the next call to this function
    // the above check will return NULL.
    if (errno == 0) {
      // do nothing
    } else {
      perror("get_next_image");
      exit(1);
    }
  }

  if (line[strlen(line) - 1] == '\n') {
    line[strlen(line) - 1] = '\0';
  }

  char *token = strtok(line, ",");
  // for error checking.
  char *endptr;
  *label = strtol(token, &endptr, 10);
  // we have not read until the end of the token
#ifndef NDEBUG
  if (*endptr != '\0') {
    fprintf(stderr, "get_next_image: invalid format.\n");
    exit(1);
  }
#endif
  vec_t *vec = new_vec(784);
  // the number of digits we have read so far.
  size_t nnums = 0;
  while ((token = strtok(NULL, ",")) != NULL) {
    (vec->data)[nnums] = (scalar_t)strtol(token, &endptr, 10);
    // so the values are between 0 and 1.
    (vec->data)[nnums] /= 255.0L;
    // we have not read until the end of the token
#ifndef NDEBUG
    if (*endptr != '\0') {
      fprintf(stderr, "get_next_image: invalid format.\n");
      exit(1);
    }
#endif
    nnums++;
  }

#ifndef NDEBUG
  // have we filled every field of the vector?
  if (nnums != vec->n) {
    fprintf(stderr, "get_next_image: invalid format.\n");
    exit(1);
  }
#endif

  return vec;
}

int main() {
  FILE *train_file = fopen("mnist_train.csv", "r");
  if (train_file == NULL) {
    perror("train_file");
    return 1;
  }

  int label;
  vec_t *i;
  nn_t *nn = new_nn(28 * 28, 30, 10);

  vec_t *o = new_vec(10);

  size_t nimg = 0;

  while ((i = get_next_image(train_file, &label)) != NULL) {
    nn_forward(o, nn, i);
    printf("++++++++++++ IMAGE NO %zu START +++++++++++++\n", nimg);
    printf("label: %d.\n", label);
    for (size_t i = 0; i < (o->n); i++)
      printf("%zu: %Lf.\n", i, (o->data)[i]);

    printf("++++++++++++ IMAGE NO %zu END +++++++++++++\n", nimg);
    free_vec(i);
    nimg++;
  }

  free_vec(o);
  free_nn(nn);

  return 0;
}
