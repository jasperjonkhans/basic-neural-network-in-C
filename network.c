#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "./matrix.h"
#include "./network.h"
#include <sys/stat.h>

// input: n * 1 vector (column vector)
static void apply(matrix * vector, float (*func)(float)){
    if(vector->cols == 1){
        for (int i = 0; i < vector->rows; i++) vector->entries[i] = func(vector->entries[i]);
    }else{
        fprintf(stderr, "apply function expects column vector, got %dx%d", vector->rows, vector->cols);
        exit(1);
    }
}

static matrix * create_one_hot(int class_index, int num_classes) {
    matrix * one_hot = matrix_create(num_classes, 1, NULL);
    for (int i = 0; i < num_classes; i++) {
        one_hot->entries[i] = 0.0;
    }
    if (class_index >= 0 && class_index < num_classes) {
        one_hot->entries[class_index] = 1.0;
    }
    return one_hot;
}

//lossfunctions
matrix * MSE(matrix * A_L, matrix * Y){
    matrix * diff = sub(A_L, Y);
    int rows = diff->rows;
    for (int i = 0; i < rows; i++){
        float error = diff->entries[i];
        diff->entries[i] = error * error;
    }
    ipscalarmul(diff, 1.0 / rows);
    return diff;
}

matrix * MSEprime(matrix * A_L, matrix * Y){
    matrix * diff = sub(A_L, Y);
    ipscalarmul(diff, 2.0 / A_L->rows);
    return diff;
}

float cross_entropy_loss(matrix * A_L, matrix * Y){
    float s = 0;
    for(int i = 0; i < Y->rows; i++){
        s += Y->entries[i] * log(A_L->entries[i]);
    }
    return -s;
}

//activationfunctions

static float float_lr(float x){
    return x > 0 ? x : 0.01 * x;
}

static float float_lr_prime(float x){
    return x > 0 ? 1 : 0.01;
}

void softmax(matrix * vector) {
  assert(vector);

  float m = -INFINITY;
  for (int i = 0; i < vector->rows; i++) {
    if (vector->entries[i] > m) {
      m = vector->entries[i];
    }
  }

  float sum = 0.0;
  for (int i = 0; i < vector->rows; i++) {
    sum += expf(vector->entries[i] - m);
  }

  float offset = m + logf(sum);
  for (int i = 0; i < vector->rows; i++) {
    vector->entries[i] = expf(vector->entries[i] - offset);
  }
}
void LeakyReLU(matrix * z){
    apply(z, float_lr);
}
void LeakyReLUprime(matrix * z){
    apply(z, float_lr_prime);
}

static float float_sigmoid(float x){
    return 1.0 / (1.0 + expf(-x));
}

static float float_sigmoid_prime(float x){
    float s = float_sigmoid(x);
    return s * (1.0 - s);
}

void sigmoid(matrix * z){
    apply(z, float_sigmoid);
}

void sigmoid_prime(matrix * z){
    apply(z, float_sigmoid_prime);
}

static void free_A(matrix ** A, int len){
    for(int i = 0; i < len; i++){
        matrix_free(A[i]);
    }
}

void test(network * net, data_set * set){
    for(int i = 0; i < set->size; i++){
        matrix ** A = NULL;
        forward(net, set->entry[i].input, &A);
        float error = cross_entropy_loss(A[net->layerc-1], set->entry[i].output);
        //printf("%f\n", error);
        backward(net, A, net->learning_rate, set->entry[i].output);
        free_A(A, net->layerc-1);
        A = NULL;
        forward(net, set->entry[i].input, &A);
        error = cross_entropy_loss(A[net->layerc-1], set->entry[i].output);
        //printf("%f\n\n", error);
    }
}



network * net_create(int input, int output, int hidden, int layerc){
    if(layerc < 1){
        fprintf(stderr, "layercount under 1 is not sensible");
        exit(1);
    }
    network * net = malloc(sizeof(network));
    net->layerc = layerc + 2;
    net->layers = malloc(net->layerc * sizeof(layer));
    net->act = malloc(sizeof(activation));
    net->act->func = LeakyReLU;
    net->act->fprime = LeakyReLUprime;
    net->loss.func = NULL;
    net->loss.fprime = NULL;
    //inputlayer 
    net->layers[0].in = 0;
    net->layers[0].out = input;
    net->layers[0].weights = NULL;
    net->layers[0].biases = NULL;
    //hiddenlayer
    for (int i = 1; i < layerc + 1; i++){
        net->layers[i].in = i == 1 ? input : hidden;
        net->layers[i].out = hidden;
        net->layers[i].weights = matrix_create(hidden, i == 1 ? input : hidden, NULL);
        net->layers[i].biases = matrix_create(hidden, 1, NULL);
        rand_init(net->layers[i].weights);
        rand_init(net->layers[i].biases);
    }
    //outputlayer
    net->layers[layerc + 1].in = hidden;
    net->layers[layerc + 1].out = output;
    net->layers[layerc + 1].weights = matrix_create(output, hidden, NULL);
    net->layers[layerc + 1].biases = matrix_create(output, 1, NULL);  
    rand_init(net->layers[layerc + 1].weights);
    rand_init(net->layers[layerc + 1].biases);
    return net;
}

void net_print(network * net){
    printf("=== NEURAL NETWORK STRUCTURE ===\n\n");
    printf("Total layers: %d (1 input + %d hidden + 1 output)\n", net->layerc, net->layerc - 2);
    printf("Activation function: %s\n", (net->act && net->act->func) ? "Assigned" : "NOT ASSIGNED");
    printf("\n");
    for (int layer = 0; layer < net->layerc; layer++) {
        const char* layer_type;
        if (layer == 0) {
            layer_type = "INPUT";
        } else if (layer == net->layerc - 1) {
            layer_type = "OUTPUT";
        } else {
            layer_type = "HIDDEN";
        }
        
        printf("--- LAYER %d (%s) ---\n", layer, layer_type);
        printf("Dimensions: %d inputs → %d outputs\n", 
               net->layers[layer].in, net->layers[layer].out);
        if (layer > 0) {
            printf("Weights:\n");
            matrix_print(net->layers[layer].weights);
            
            printf("Biases:\n");
            matrix_print(net->layers[layer].biases);
        } else {
            printf("  No weights/biases (input layer)\n");
        }
        printf("\n");
    }
    
    printf("=== END OF NETWORK ===\n");
}

matrix *forward(network *net, matrix *input, matrix ***A) {
    const int L = net->layerc;
    assert(input->cols==1);

    *A = malloc((L) * sizeof(matrix*));
    if (!*A) return NULL;

    (*A)[0] = copy(input);

    matrix *vector = copy(input);

    for (int i = 1; i < L; i++) {
        matrix *old_vector = vector;
        vector = dot(net->layers[i].weights, vector);
        matrix_free(old_vector);

        ipadd(vector, net->layers[i].biases);

        if (i < L - 1 && net->act && net->act->func) {
            net->act->func(vector);
        }

        (*A)[i] = copy(vector);
    }

    softmax((*A)[L-1]);
    return (*A)[L-1];
}

float percentage(matrix * A_L, matrix * Y) {
    for(int i = 0; i < Y->rows; i++){
        if(Y->entries[i] == 1){
            return 100 * A_L->entries[i];
        }
    }
}

void net_save(network * net){
    
}


void net_train(network * net, int epochs, data_set * set, float learningrate){
    float percent = 0;
    float sum = 0;
    int k = 0;
    printf("neural net training...\n\n");
    if(!set){ printf("set is null"); exit(1); }
    int *indices = malloc((size_t)set->size * sizeof *indices);
    if (!indices) { perror("malloc"); exit(1); }
    for (int i = 0; i < set->size; i++) indices[i] = i; 
    for (int i = 0; i < epochs; i++){
        shuffle(indices, set->size);
        for(int j = 0; j < set->size; j++) {
            matrix **A = NULL;
            matrix *out = forward(net, set->entry[indices[j]].input, &A);
            sum += percentage(out, set->entry[indices[j]].output);
            if(j%((int)(set->size * epochs/1000)) == 0){
                float acc = (sum / k - 0.1f)*(10/9) ;
                printf("\r%.1f%% progress | %.1f%% accuracy", percent, acc);
                percent += 0.1f;
                fflush(stdout);
                k = 0;
                sum = 0;
            }
            k++;
            //net_print(net);
            backward(net, A, learningrate, set->entry[indices[j]].output);
            // free A[0..L-1] (Input + all layers)
            for (int k = 0; k < net->layerc; k++) matrix_free(A[k]);
            free(A);
        }
    }
    free(indices);
    printf("\n\ntraining complete!\n");
}

float accuracy(network * net, data_set * set){
    float s = 0;
    for(int i = 0; i < set->size; i++){
        matrix ** A = NULL;
        matrix * out = forward(net, set->entry[i].input, &A);
        s += cross_entropy_loss(out, set->entry[i].output);
        for(int j = 0; j < net->layerc; j++){
            matrix_free(A[j]);
        }
        if(out) matrix_free(out);
    }
    return s / set->size;
}

void backward(network *net, matrix **A, float lr, matrix *Y) {
    int L = net->layerc - 1;
    matrix *A_L    = A[L];
    matrix *A_prev = A[L-1];
    matrix *delta  = sub(A_L, Y);
    matrix *Aprev_T= transpose(A_prev);
    matrix *dW     = dot(delta, Aprev_T);
    matrix *scaled_delta = scalarmul(delta, lr);
    ipscalarmul(dW, lr);
    ipsub(net->layers[L].weights, dW);
    ipsub(net->layers[L].biases,  scaled_delta);
    matrix_free(scaled_delta);
    matrix_free(Aprev_T);
    matrix_free(dW);
    matrix *prev_delta = delta;
    for (int l = L-1; l >= 1; --l) {
        matrix *W_up_T = transpose(net->layers[l+1].weights);
        matrix *err_l    = dot(W_up_T, prev_delta);
        matrix_free(prev_delta);
        matrix_free(W_up_T);
        matrix *prime_l  = copy(A[l]);
        net->act->fprime(prime_l);
        matrix *delta_l  = hadamard_product(err_l, prime_l);
        matrix_free(err_l);
        matrix_free(prime_l);
        matrix *A_down_T  = transpose(A[l-1]);
        matrix *dW_l     = dot(delta_l, A_down_T);
        matrix *scaled_delta_l = scalarmul(delta_l, lr);
        ipscalarmul(dW_l, lr);
        ipsub(net->layers[l].weights, dW_l);
        ipsub(net->layers[l].biases,  scaled_delta_l);
        matrix_free(scaled_delta_l);
        matrix_free(A_down_T);
        matrix_free(dW_l); 
        prev_delta = delta_l;
    }
    matrix_free(prev_delta);
}

