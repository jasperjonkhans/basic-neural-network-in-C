#include <stdio.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    double **entries;
    int rows;
    int cols;
} matrix;

typedef struct {
    double *entries;
    int rows;
} vector;

matrix *create_matrix(int rows, int cols);
matrix *dot_product(matrix * m1, matrix * m2);
matrix *add(matrix * m1, matrix * m2);
matrix *subtract(matrix * m1, matrix * m2);
matrix *transpose(matrix * m);

#endif
