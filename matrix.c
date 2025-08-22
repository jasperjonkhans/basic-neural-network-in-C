#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./matrix.h"

typedef struct {
    double **entries;
    int rows;
    int cols;
} matrix; // lowercase because uppercase is tideous to type

matrix * create_matrix(int rows, int cols){
    matrix * m;
    m->rows = rows;
    m->cols = cols;
    m->entries = malloc(sizeof(double *));
    for (int i = 0; i < rows; i++){
        m->entries[i] = malloc(sizeof(double));
    }
    return m;
}

matrix * dot_product(matrix * m1, matrix * m2){
    if (m1->cols == m2->rows){
        matrix * result = create_matrix(m1->rows, m2->cols);
        for (int i = 0; i < m1->rows; i++){
            for (int j = 0; j < m2->cols; j++){
                double sum = 0;
                for (int k = 0; k < m1->cols; k++){
                    sum += m1->entries[i][k] * m2->entries[k][j];
                }
            }
        }
        return result;
    }else{
        fprint(stderr, "dimensions unfit for matrix multiplication, %dx%d, %dx%d");
        exit(1);
    }
}

matrix * add(matrix * m1, matrix * m2){
    if(m1->cols == m2->cols && m1->rows == m2->rows){
        matrix * result = create_matrix(m1->rows, m1->cols);
        for(int i = 0; i < m1->rows; i++){
            for(int j = 0; j < m1->cols; j++){
                result->entries[i][j] = m1->entries[i][j] + m2->entries[i][j];
            }
        }
        return result;
    }else{
        fprint(stderr, "dimensions unfit for matrix multiplication, %dx%d, %dx%d");
        exit(1);
    }
}

matrix * subtract(matrix * m1, matrix * m2){
    if(m1->cols == m2->cols && m1->rows == m2->rows){
        matrix * result = create_matrix(m1->rows, m1->cols);
        for(int i = 0; i < m1->rows; i++){
            for(int j = 0; j < m1->cols; j++){
                result->entries[i][j] = m1->entries[i][j] - m2->entries[i][j];
            }
        }
        return result;
    }else{
        fprint(stderr, "dimensions unfit for matrix multiplication, %dx%d, %dx%d");
        exit(1);
    }
}

matrix * transpose(matrix * m){
    matrix * result = create_matrix(m->cols, m->rows);
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            result->entries[j][i] = m->entries[i][j];
        }
    }
    return result;
}

void rand_init(matrix * m){
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            m->entries[i][j] = he_init(m->rows)
        }
    }
}

void matrix_free(matrix *m) {
	for (int i = 0; i < m->rows; i++) {
		free(m->entries[i]);
	}
	free(m->entries);
	free(m);
	m = NULL;
}

double he_init(int in_degree){
    double limit = sqrt(6.0/in_degree);
    return ((double) rand() / (double) RAND_MAX * 2 - 1) * limit;
}