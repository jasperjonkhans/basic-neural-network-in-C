#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./matrix.h"

typedef float (*act_f)(float);
typedef float (*act_fprime)(float);
typedef struct {
    act_f func;
    act_fprime fprime;
} activation;

typedef struct {
    int in;
    int out;
    matrix * weights; // dim: in * out
    matrix * biases;  // dim: out * 1
} layer;

typedef struct {
    int layerc;
    layer * layers;
    activation * act;
    float (*loss_function)(float);
} network;

typedef struct {
    matrix * img;
} data_entry;

// activation functions

inline float LeakyReLU(float signal){
    return signal > 0 ? signal : 0.01 * signal;
}

inline float LeakyReLU_prime(float signal){
    return signal > 0 ? 1 : 0.01;
}

inline float ReLU(float signal){
    return signal > 0 ? signal : 0;
}

inline float ReLU_prime(float signal){
    return signal > 0 ? 1 : 0;
}

// loss functions

inline float cross_entropy_loss(matrix * v, matrix * y){
    float sum = 0;
    for (int i = 0; i < v->rows; i++){
        sum += y->entries[i] * log(v->entries[i]);
    }
    return -sum;
}

network * net_create(int input, int output, int hidden, int layerc);
network * net_load(char * location_string);
matrix * net_predict(network * net, matrix * input);
void free_snapshot(matrix ** snap, int layerc);
void net_train(network * net);
void net_save(network * net);
void net_free(network * net);
void net_print(network * net);
void apply(matrix * m, float (*func)(float));
double matrix_get(matrix * m, int row, int col);
void matrix_set(matrix * m, int row, int col, double value);
double mean_squared_error(matrix * output, matrix * training);