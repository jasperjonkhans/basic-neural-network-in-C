#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "./matrix.h"
#include "./network.h"

float lossfunc(matrix * predicted, matrix * training, float (*func)(matrix *, matrix *)){
    return func(predicted, training);
}

void actfunc(matrix * v, network * net){
    if (net->act && net->act->func) {
        apply(v, net->act->func);
    }
}

network * net_create(int input, int output, int hidden, int layerc){

    if(layerc < 1){
        fprintf(stderr, "layercount under 1 is not sensible");
        exit(1);
    }

    network * net = malloc(sizeof(network));
    net->layerc = layerc + 2; // layerc hidden layers + input layer + output layer
    net->layers = malloc(net->layerc * sizeof(layer));
    
    // Initialize activation function to NULL (should be set later)
    net->act = NULL;
    net->loss_function = NULL;
    
    //inputlayer 
    net->layers[0].in = 0;
    net->layers[0].out = input;
    net->layers[0].weights = NULL;
    net->layers[0].biases = NULL;

    //hiddenlayer
    for (int i = 1; i < layerc + 1; i++){
        net->layers[i].in = i == 1 ? input : hidden;
        net->layers[i].out = hidden;
        net->layers[i].weights = matrix_create(hidden, i == 1 ? input : hidden);
        net->layers[i].biases = matrix_create(hidden, 1);
        rand_init(net->layers[i].weights);
        rand_init(net->layers[i].biases);
    }

    //outputlayer
    net->layers[layerc + 1].in = hidden;
    net->layers[layerc + 1].out = output;
    net->layers[layerc + 1].weights = matrix_create(output, hidden);
    net->layers[layerc + 1].biases = matrix_create(output, 1);  
    rand_init(net->layers[layerc + 1].weights);
    rand_init(net->layers[layerc + 1].biases);
    
    return net;
}

void net_print(network * net){
    printf("=== NEURAL NETWORK STRUCTURE ===\n\n");
    
    // Netzwerk Übersicht
    printf("Total layers: %d (1 input + %d hidden + 1 output)\n", net->layerc, net->layerc - 2);
    printf("Activation function: %s\n", (net->act && net->act->func) ? "Assigned" : "NOT ASSIGNED");
    printf("\n");
    
    // Detaillierte Layer-Informationen
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
        
        // Skip weights/biases for input layer (they are NULL)
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

void net_free(network * net) {
    for (int i = 0; i < net->layerc; i++) {
        // Skip input layer (has NULL weights/biases)
        if (i > 0) {
            matrix_free(net->layers[i].weights);
            matrix_free(net->layers[i].biases);
        }
    }
    free(net->layers);
    free(net);
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

matrix * net_predict(network * net, matrix * input){
    matrix * vector = copy(input);
    matrix **snapshot = malloc(sizeof(matrix*) * net->layerc);
    
    for (int i = 1; i < net->layerc; i++){
        matrix * old_vector = vector;
        vector = dot_product(net->layers[i].weights, vector);
        matrix_free(old_vector);
        
        add(vector, net->layers[i].biases);
        
        if (i < net->layerc - 1 && net->act && net->act->func) {
            apply(vector, net->act->func);
        }
        snapshot[i-1] = copy(vector);
    }
    
    
    free_snapshot(snapshot, net->layerc - 1);
    softmax(vector);
    return vector;
}

void free_snapshot(matrix ** snapshot, int layerc){
    for (int i = 0; i < layerc - 1; i++) {
        matrix_free(snapshot[i]);
    }
    free(snapshot);
}

// input: n * 1 vector (column vector)
void apply(matrix * vector, float (*func)(float)){
    if(vector->cols == 1){
        for (int i = 0; i < vector->rows; i++){
            vector->entries[i] = func(vector->entries[i]);
        }
    }else{
        fprintf(stderr, "apply function expects column vector, got %dx%d", vector->rows, vector->cols);
        exit(1);
    }
}