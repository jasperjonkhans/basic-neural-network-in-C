#include "./matrix.h"
#include "./network.h"
#include <stdio.h>

#define NumInputNeurons 4
#define NumOutputNeurons 4
#define NumHiddenNeurons 4
#define NumHiddenLayers 4


int main(){
    network * neural_net = net_create(NumInputNeurons, NumOutputNeurons, NumHiddenNeurons, NumHiddenLayers);
    
    // Create column vector (4x1), not row vector
    matrix * input = matrix_create(NumInputNeurons, 1);
    
    // Initialize input with some values for testing
    for(int i = 0; i < NumInputNeurons; i++) {
        input->entries[i] = 0.5; // Set all inputs to 0.5
    }
    
    matrix * output = net_predict(neural_net, input);
    printf("Network output:\n");
    matrix_print(output);
    
    // Cleanup
    matrix_free(input);
    matrix_free(output);
    net_free(neural_net);
    
    return 0;
}