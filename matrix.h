#include <stdio.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    double *entries; // 1d for performance gains
    int rows;
    int cols;
} matrix; // lowercase because uppercase is tideous to type

matrix *matrix_create(int rows, int cols);
matrix *dot_product(matrix * m1, matrix * m2);
matrix *copy(matrix * m);
matrix *transpose(matrix * m);
matrix *transposed_vector_projection(matrix * v, matrix * m);
void add(matrix * m1, matrix * m2);
void subtract(matrix * m1, matrix * m2);
void rand_init(matrix * m);
void matrix_free(matrix * m);
void matrix_set(matrix * m, int row, int col, double value);
void matrix_print(matrix * m);
double he_init(int in_degree);
double matrix_get(matrix * m, int row, int col);

#endif