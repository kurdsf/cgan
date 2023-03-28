#include "nn.h"
#include "utest.h"
#include <gsl/gsl_matrix.h>
#include <stdio.h>

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

UTEST_MAIN();
