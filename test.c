#include "nn.h"
#include "utest.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>

/* This has to be refactored to use GPL later on.
#define TRAIN_SIZE 1000
#define MAX_ERR 0.01L

// check if the neural network is able to learn the binary operator op.
// return the average error rate after some training.
static scalar_t learn_bin_op(int (*op)(int, int)) {
  srand(time(NULL));
  nn_t *nn = new_nn(2, 4, 1);
  for (size_t i = 0; i < TRAIN_SIZE; i++) {
    // generate a either 0 or 1.
    vec_t *data = new_vec(2);
    vec_t *label = new_vec(1);
    int a = rand() % 2;
    int b = rand() % 2;
    (data->data)[0] = a;
    (data->data)[1] = b;
    (label->data)[0] = op(a, b);
    nn_train(nn, data, label);
    free_vec(data);
    free_vec(label);
  }

  // after we have trained our neural network,
  // compute the average error
  scalar_t avg_err = 0.0F;
  assert(TRAIN_SIZE >= 10);
  for (size_t i = 0; i < 10; i++) {
    // generate a either 0 or 1.
    vec_t *data = new_vec(2);
    vec_t *label = new_vec(1);
    int a = rand() % 2;
    int b = rand() % 2;
    (data->data)[0] = a;
    (data->data)[1] = b;
    (label->data)[0] = op(a, b);
    avg_err += nn_train(nn, data, label);
    free_vec(data);
    free_vec(label);
  }
  free_nn(nn);
  return avg_err;
}

static int ixor(int a, int b) { return a ^ b; }
static int iand(int a, int b) { return a && b; }

UTEST(nn, nn_xor) {
  scalar_t err = learn_bin_op(&ixor);
  ASSERT_LT(err, MAX_ERR);
}

UTEST(nn, nn_and) {
  scalar_t err = learn_bin_op(&iand);
  ASSERT_LT(err, MAX_ERR);
}

UTEST_MAIN();
*/
