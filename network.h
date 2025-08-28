#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./matrix.h"

typedef void (*act_f)(matrix *);
typedef void (*act_fprime)(matrix *);
typedef matrix * (*loss_f)(matrix *, matrix *);
typedef matrix * (*loss_fprime)(matrix *, matrix *);
typedef struct {
    act_f func;
    act_fprime fprime;
} activation;
typedef struct {
    loss_f func;
    loss_fprime fprime;
} loss;

typedef struct {
    int in;
    int out;
    matrix * weights; // dim: out * in
    matrix * biases;  // dim: out * 1
} layer;

typedef struct {
    int layerc;
    float learning_rate;
    layer * layers;
    activation * act;
    loss loss;
} network;

typedef struct {
    matrix * input;
    matrix * output;
} data_point;

typedef struct {
    int size;
    data_point * entry;
} data_set;

//neural net main funcs
float accuracy(network * net, data_set * set);
void test(network * net, data_set * set);
network * net_create(int input, int output, int hidden, int layerc);
matrix * forward(network * net, matrix * input, matrix *** A);
void backward(network *net, matrix **A, float lr, matrix *Y);
void net_print(network * net);
void net_train(network * net, int epochs, data_set * set, float learningrate);
//lossfunctions
matrix * MSE(matrix * output, matrix * expected);
matrix * MSEprime(matrix * output, matrix * expected);
//activationfunctions
void softmax(matrix * m);
void LeakyReLU(matrix * z);
void LeakyReLUprime(matrix * z);