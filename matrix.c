#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./matrix.h"

matrix * matrix_create(int rows, int cols){
    matrix * m = malloc(sizeof(matrix));
    m->rows = rows;
    m->cols = cols;
    m->entries = malloc(rows * cols * sizeof(double));  
    return m;
}

matrix *dot_product(matrix * m1, matrix * m2){
    // standard matrix multiplication: m1 (a x b) * m2 (b x c) = result (a x c)
    if(m1->cols == m2->rows){
        matrix * result = matrix_create(m1->rows, m2->cols);
        for(int i = 0; i < m1->rows; i++){
            for(int j = 0; j < m2->cols; j++){
                double sum = 0;
                for(int k = 0; k < m1->cols; k++){
                    sum += m1->entries[i * m1->cols + k] * m2->entries[k * m2->cols + j];
                }
                result->entries[i * result->cols + j] = sum;
            }
        }
        return result;
    }else{
        fprintf(stderr, "dimensions unfit for matrix multiplication, %dx%d * %dx%d", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

void add(matrix * m1, matrix * m2){
    if(m1->cols == m2->cols && m1->rows == m2->rows){
        for(int i = 0; i < m1->rows; i++){
            for(int j = 0; j < m1->cols; j++){
                m1->entries[i * m1->cols + j] += m2->entries[i * m2->cols + j];
            }
        }
    }else{
        fprintf(stderr, "dimensions unfit for matrix addition, %dx%d, %dx%d", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

void subtract(matrix * m1, matrix * m2){
    if(m1->cols == m2->cols && m1->rows == m2->rows){
        for(int i = 0; i < m1->rows; i++){
            for(int j = 0; j < m1->cols; j++){
                m1->entries[i * m1->cols + j] -= m2->entries[i * m2->cols + j];
            }
        }
    }else{
        fprintf(stderr, "dimensions unfit for matrix subtraction, %dx%d, %dx%d", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

matrix * transpose(matrix * m){
    matrix * result = matrix_create(m->cols, m->rows);
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            result->entries[j * result->cols + i] = m->entries[i * m->cols + j];
        }
    }
    return result;
}

void rand_init(matrix * m){
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            m->entries[i * m->cols + j] = he_init(m->rows);
        }
    }
}

void matrix_free(matrix *m) {
    if (m != NULL) {  // NULL-Check hinzufügen
        if (m->entries != NULL) {
            free(m->entries);
        }
        free(m);
    }
}

double he_init(int in_degree){
    double limit = sqrt(6.0/in_degree);
    return ((double) rand() / (double) RAND_MAX * 2 - 1) * limit;
}

double matrix_get(matrix * m, int row, int col) {
    return m->entries[row * m->cols + col];
}

void matrix_set(matrix * m, int row, int col, double value) {
    m->entries[row * m->cols + col] = value;
}

matrix * copy(matrix * m){
    matrix * clone = matrix_create(m->rows, m->cols);
    // Kopiere die Daten, nicht nur den Zeiger
    for (int i = 0; i < m->rows * m->cols; i++) {
        clone->entries[i] = m->entries[i];
    }
    return clone;
}

void matrix_print(matrix * m) {
    if (m == NULL) {
        printf("  NULL matrix\n");
        return;
    }
    
    //printf("  Matrix (%dx%d):\n", m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        printf("  [");
        for (int j = 0; j < m->cols; j++) {
            printf(" %8.4f", m->entries[i * m->cols + j]);
        }
        printf(" ]\n");
    }
}