#include<assert.h>
#include<errno.h>
#include<time.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include"mat.h"
#include"nn.h"


static void vec_softmax(vec_t* x);

nn_t* new_nn(size_t isize, size_t hsize, size_t osize) {
        nn_t* res = malloc(sizeof(nn_t));
#ifndef NDEBUG
        if(res == NULL) {
                perror("new_nn");
                exit(1);
        }
#endif

        res->w1 = new_mat(hsize, isize);
        res->w2 = new_mat(osize, hsize);


        srand48(time(NULL));

        // intialize w1 and w2 with random values from [0.0, 1.0).
        // you may want to check out `man drand64`.
        for(size_t i = 0; i < (res->w1->m)*(res->w1->n); i++) {
                (res->w1->data)[i] = (scalar_t) drand48();
        }
        for(size_t i = 0; i < (res->w2->m)*(res->w2->n); i++) {
                (res->w2->data)[i] = (scalar_t) drand48();
        }


        return res;
}


void nn_forward(vec_t* dest, const nn_t* nn, vec_t* input) {
#ifndef NDEBUG
        assert(dest->n == nn->w2->m);
#endif

        // hiden layer
        vec_t* h = new_vec(nn->w1->m);
        mat_vec_mul(h, nn->w1, input);

        // apply RELU
        for(size_t i=0; i < (h->n); i++) {
                if((h->data)[i] < 0.0) {
                        (h->data)[i] = 0.0;
                }
        }

        // output layer
        mat_vec_mul(dest, nn->w2, h);
        free_vec(h); // no longer needed.


        vec_softmax(dest);
}

// numerically stable softmax
// approach taken from:
// https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
void vec_softmax(vec_t* x) {
        // find the maximum of x
        scalar_t max = (x->data)[0];
        for(size_t i=1; i < (x->n); i++) {
                if((x->data)[i] > max) max = (x->data)[i];
        }

        for(size_t i=0; i < (x->n); i++) {
                (x->data)[i] -= max;
        }

        scalar_t sum = 0;
        for(size_t i=0; i < (x->n); i++) {
                sum += expl((x->data)[i]);
#ifndef NDEBUG
                if(errno != 0) {
                        perror("softmax");
                        exit(1);
                }
#endif 
        }


        for(size_t i=0; i < (x->n); i++) {
                (x->data)[i] = expl((x->data)[i]) / sum;
#ifndef NDEBUG
                if(errno != 0) {
                        perror("softmax");
                        exit(1);
                }
#endif
        }

}


void free_nn(nn_t* nn) {
        free_mat(nn->w1);
        free_mat(nn->w2);
        free(nn);
}



