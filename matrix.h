#include <stdio.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float *entries;
    int rows;
    int cols;
} matrix;

//management
matrix * copy(matrix * m);
void matrix_print(matrix * m);
void shuffle(int * arr, int len);
void matrix_flatten(matrix * m);
void matrix_free(matrix * m);
void rand_init(matrix *m);
matrix * matrix_create(int rows, int cols, float * entries);
//not in place
matrix * dot(matrix * m1, matrix * m2);
matrix * transpose(matrix * m);
matrix * add(matrix * m1, matrix * m2);
matrix * sub(matrix * m1, matrix * m2);
matrix * hadamard_product(matrix * m1, matrix * m2);
matrix * scalarmul(matrix * m, float scalar);
//in place 
void ipadd(matrix * m1, matrix * m2);
void ipsub(matrix * m1, matrix * m2);
void ipscalarmul(matrix * m, float scalar);
float sum(matrix * m);


#endif