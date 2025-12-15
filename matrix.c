#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./matrix.h"

static float matrix_get(matrix * m, int row, int col) {
    return m->entries[row * m->cols + col];
}
static void matrix_set(matrix * m, int row, int col, float value) {
    m->entries[row * m->cols + col] = value;
}
matrix * copy(matrix * m){
    matrix * clone = matrix_create(m->rows, m->cols, m->entries);
    return clone;
}
static matrix * add_scalar(matrix * m, float scalar){
    matrix * result = matrix_create(m->rows, m->cols, NULL);
    int len = m->rows * m->cols;
    for(int i = 0; i < len; i++){
        result->entries[i] = m->entries[i] + scalar;
    }
    return result;
}
static matrix * subtract_scalar(matrix * m, float scalar){
    matrix * result = matrix_create(m->rows, m->cols, NULL);
    int len = m->rows * m->cols;
    for(int i = 0; i < len; i++){
        result->entries[i] = m->entries[i] - scalar;
    }
    return result;
}


static float he_init(int in_degree){
    double limit = sqrt(6.0/in_degree);
    return ((double) rand() / (double) RAND_MAX * 2 - 1) * limit;
}

void rand_init(matrix *m){
    int len = m->rows * m->cols;
    for(int i = 0; i < len; i++){
        m->entries[i] = he_init(m->rows);
    }
}
//management funcs

void shuffle(int * arr, int len){
    for(int i = len - 1; i > 0; i--){
        int j = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
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

void matrix_free(matrix *m) {
    if (m != NULL) {
        if (m->entries != NULL) {
            free(m->entries);
        }
        free(m);
    }
}
matrix * matrix_create(int rows, int cols, float * entries){
    matrix * m = malloc(sizeof(matrix));
    m->rows = rows;
    m->cols = cols;
    m->entries = malloc(sizeof(float) * rows * cols);
    if(entries){ 
        for(int i = 0; i < cols * rows; i++) m->entries[i] = entries[i];
    }
    return m;
}

void matrix_flatten(matrix * m){
    m->rows = m->rows * m->cols;
    m->cols = 1;
}
//not in place

matrix * scalarmul(matrix * m, float scalar){
    matrix * clone = copy(m);
    ipscalarmul(clone, scalar);
    return clone;
}
matrix *dot(matrix * m1, matrix * m2){
    // standard matrix multiplication: m1 (a x b) * m2 (b x c) = result (a x c)
    if(m1->cols == m2->rows){
        matrix * result = matrix_create(m1->rows, m2->cols, NULL);
        for(int i = 0; i < m1->rows; i++){
            for(int j = 0; j < m2->cols; j++){
                float sum = 0;
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
matrix * transpose(matrix * m){
    matrix * result = matrix_create(m->cols, m->rows, NULL);
    for(int i = 0; i < m->rows; i++){
        for(int j = 0; j < m->cols; j++){
            result->entries[j * result->cols + i] = m->entries[i * m->cols + j];
        }
    }
    return result;
}
matrix * add(matrix * m1, matrix * m2){
    matrix * temp = copy(m1);
    ipadd(temp, m2);
    return temp;
}
matrix * sub(matrix * m1, matrix * m2){
    matrix * temp = copy(m1);
    ipsub(temp, m2);
    return temp;
}
matrix * hadamard_product(matrix * m1, matrix * m2){
    if(m1->cols == m2->cols && m1->rows == m2->rows){
        matrix * result = matrix_create(m1->rows, m1->cols, NULL);
        int len = m1->rows * m1->cols;
        for(int i = 0; i < len; i++){
            result->entries[i] = m1->entries[i] * m2->entries[i];
        }
        return result;
    }else{
        fprintf(stderr, "dimensions unfit for taking the hadamard product, %dx%d, %dx%d", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}
//in place 
float sum(matrix * m){
    float s = 0;
    for(int i = 0; i < m->rows * m->cols; i++){
        s += m->entries[i];
    }
    return s;
}
void ipadd(matrix * m1, matrix * m2){
    if(m1->cols == m2->cols && m1->rows == m2->rows){
        int len = m1->rows * m1->cols;
        for(int i = 0; i < len; i++){
            m1->entries[i] += m2->entries[i];
        }
    }else{
        fprintf(stderr, "dimensions unfit for matrix addition, %dx%d, %dx%d", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}

void ipsub(matrix * m1, matrix * m2){
    if(m1->cols == m2->cols && m1->rows == m2->rows){
        int len = m1->rows * m1->cols;
        for(int i = 0; i < len; i++){
            m1->entries[i] -= m2->entries[i];
        }
    }else{
        fprintf(stderr, "dimensions unfit for matrix subtraction, %dx%d, %dx%d", m1->rows, m1->cols, m2->rows, m2->cols);
        exit(1);
    }
}
void ipscalarmul(matrix * m, float scalar){
    int len = m->rows * m->cols;
    for(int i = 0; i < len; i++){
        m->entries[i] *= scalar;
    }
}