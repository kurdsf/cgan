#include<assert.h>
#include<math.h>
#include<stdio.h>
#include<stdlib.h>

#include"mat.h"
#include"nn.h"



nn_t* new_nn(size_t isize, size_t hsize, size_t osize) {
        nn_t* res = malloc(sizeof(nn_t));
        if(res == NULL) {
                perror("new_nn");
                exit(1);
        }

        res->w1 = new_mat(hsize, isize);
        res->w2 = new_mat(osize, hsize);

        // intialize w1 and w2 with random values from [0.0, 1.0).
        // you may want to check out `man drand64`.
        for(size_t i = 0; i < (res->w1->m)*(res->w1->n); i++) {
                (res->w1->data)[i] = drand48();
        }
        for(size_t i = 0; i < (res->w2->m)*(res->w2->n); i++) {
                (res->w2->data)[i] = drand48();
        }

        return res;
}


void nn_forward(vec_t* dest, const nn_t* nn, vec_t* input) {
        assert(dest->n == nn->w2->m);

        // hiden layer
        vec_t* h = new_vec(nn->w1->m);
        mat_vec_mul(h, nn->w1, input);

        // apply sigmoid
        for(size_t i=0; i < (h->n); i++) {
                (h->data)[i] = 1.0 / (1.0 + exp(-((h->data)[i])));
        }

        // output layer
        mat_vec_mul(dest, nn->w2, h);
        free_vec(h); // no longer needed.

        // apply softmax
        double sum = 0;
        for(size_t i=0; i < (dest->n); i++) {
                sum += exp((dest->data)[i]);
        }


        for(size_t i=0; i < (dest->n); i++) {
                (dest->data)[i] = exp((dest->data)[i]) / sum;
        }

}


void free_nn(nn_t* nn) {
        free_mat(nn->w1);
        free_mat(nn->w2);
        free(nn);
}



