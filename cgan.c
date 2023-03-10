#include<errno.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include"mat.h"
#include"nn.h"



// Returns NULL incase of EOF.
// Furthermore, it expects '\n' instead of '\r\n'
vec_t* get_next_image(FILE* f, int* label) {
        char* line = NULL;
        size_t n = 0;
        ssize_t errc = getline(&line, &n, f);
        if(errc == (ssize_t) -1) {
                // We reached EOF 
                if(errno == 0) {
                        return NULL;
                }
                perror("get_next_image");
                exit(1);
        }

        // remove the newline character
        // as it makes processing easier.
        // notice that errc is also the number
        // of chars read.
        if(line[errc - 1] == '\n') {
                line[errc - 1] = '\0';
        }

        char* token = strtok(line, ",");
        // for error checking.
        char* endptr;
        *label = (double) strtol(token, &endptr, 10);
        // we have not read until the end of the token
        if(*endptr != '\0') {
                puts("@1");
                fprintf(stderr, "get_next_image: invalid format.\n");
                exit(1);
        }
        vec_t* vec = new_vec(784);
        // the number of digits we have read so far.
        size_t nnums = 0;
        while((token = strtok(NULL, ",")) != NULL) {
                (vec->data)[nnums] = (double) strtol(token, &endptr, 10);
                // we have not read until the end of the token
                if(*endptr != '\0') {
                        printf("endptr: %c, %d", *endptr, *endptr);
                        fprintf(stderr, "get_next_image: invalid format.\n");
                        exit(1);
                }
                nnums++;
                        
        }
        
        // have we filled every field of the vector?
        if(nnums != vec->n) {
                fprintf(stderr, "get_next_image: invalid format.\n");
                exit(1);
        }

        free(line);

        return vec;

}

int main() {
        FILE* train_file = fopen("mnist_train.csv", "r");
        if(train_file == NULL) {
                perror("train_file");
                return 1;
        }

        int label;
        vec_t* i = get_next_image(train_file, &label);

        nn_t* nn = new_nn(i->n, 30, 10);

        vec_t* o = new_vec(10);

        nn_forward(o, nn, i);
        for(size_t i=0; i < (o->n); i++) 
                printf("%zu: %f.\n", i, (o->data)[i]);
       
        free_nn(nn);
        free_vec(o);
        free_vec(i);

        return 0;
}

