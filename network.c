#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./matrix.h"

typedef struct {
    int input;
    int output;
    int hidden;
    int hidden_layers;
    double learning_rate;
    matrix **hidden_weights;
    matrix **hidden_biases;
    matrix *output_weights;
    matrix *output_biases;
} network;


network * net_create(int input, int output, int hidden, double learning_rate);
network * net_load(char * location_string);
matrix * net_predict(network * net);
void net_train(network * net);
void net_save(network * net);
double LeakyReLU(double activation);
float loss(matrix * predicted, matrix * training, float (*func)(matrix *, matrix *));

float loss(matrix * predicted, matrix * training, float (*func)(matrix *, matrix *)){
    return func(predicted, training);
}

double mean_squared_error(matrix * predicted, matrix * data){
    double sum = 0;
    matrix *m = subtract(predicted, data);
    for (int i = 0; i < m->rows; i++){
        sum += m->entries[i][0] * m->entries[i][0];
    }
    return sum;
}

double LeakyReLU(double activation){
    return activation > 0 ? activation : 0.01 * activation;
}





network * net_create(int input, int output, int hidden, double learning_rate){
    network * net = {input, output, hidden, learning_rate, NULL, NULL};
    for(int i = 0; i < net->hidden_layers; i++){
        net->hidden_weights[i] = create_matrix(hidden, hidden);
    }
    for (int i = 0; i < output; i++){
        for (int j = 0; j < output; j++){
            net->output_weights->entries[i][j] = he_init(output);
        }
    }
    return net;
}