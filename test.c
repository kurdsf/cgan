#include<stdlib.h>
#include"utest.h"
#include"mat.h"
#include"nn.h"

UTEST(mat_h, vec_mat_mul) {
        for(size_t i=0; i < 100; i++) {
                vec_t* x = new_vec(i);
                mat_t* A = new_mat(i, i);
                vec_t* b = new_vec(i);
                for(size_t j = 0; j < (x->n); j++) {
                        (x->data)[j] = (scalar_t) drand48();
                        for(size_t k = 0; k < (A->n); k++) {
                                if(k == j) {
                                        (A->data)[(A->m) * j + k] = 1.0L;
                                } else {
                                        (A->data)[(A->m) * j + k] = 0.0L;
                                }
                        }
                }

                mat_vec_mul(b, A, x);

                for(size_t j = 0; j < (b->n); j++) {
                        ASSERT_EQ((b->data)[j], (x->data)[j]);
                }


                free_vec(x);
                free_mat(A);
                free_vec(b);
        }

        vec_t* x = new_vec(3);
        mat_t* A = new_mat(2, 3);
        vec_t* b = new_vec(2);

        (x->data)[0] = 3.0L;
        (x->data)[1] = 2.1L;
        (x->data)[2] = 0.0L;

        (A->data)[2 * 0 + 0] = 1.2L;
        (A->data)[2 * 0 + 1] = 0.4L;
        (A->data)[2 * 1 + 0] = 0.0L;
        (A->data)[2 * 1 + 1] = 0.2L;
        (A->data)[2 * 2 + 0] = -1.0L;
        (A->data)[2 * 2 + 1] = 3.3L;

        mat_vec_mul(b, A, x);

        ASSERT_EQ((b->data)[0], );
        ASSERT_EQ((b->data)[1], );


        free_vec(x);
        free_mat(A);
        free_vec(b);
}

UTEST_MAIN();

 
